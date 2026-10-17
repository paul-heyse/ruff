//! Owned, bounded lexical observations from Ruff's existing Checker traversal.
//!
//! IDs are local to one invocation; ranges are source bytes. Node callbacks capture the active
//! context, while binding/reference rows are emitted only after deferred passes. Ruff
//! lexical qualified names are not filesystem or instance-member resolution. A failed invocation
//! leaves only partial sink output: consumers must publish it as incomplete or discard it.

use std::path::{Path, PathBuf};

use ruff_python_ast::{AnyNodeRef, Expr, NodeKind, PySourceType, PythonVersion};
use ruff_python_codegen::Stylist;
use ruff_python_index::Indexer;
use ruff_python_parser::{ParseOptions, Parsed, parse_unchecked};
use ruff_python_semantic::{
    BindingKind, Definition, ExecutionContext, Imported, ScopeKind, SemanticModel,
};
use ruff_text_size::{Ranged, TextRange};

use crate::Locator;
use crate::checkers::ast::{LintContext, check_ast_with_observation};
use crate::noqa::NoqaMapping;
use crate::package::PackageRoot;
use crate::settings::{LinterSettings, TargetVersion, flags};

/// Every fact-affecting input is explicit. No config files, imports or environment are discovered.
/// Platform is recorded, not evaluated: Ruff's lexical traversal does not fold platform branches.
#[derive(Debug, Clone)]
pub struct Settings {
    pub path: PathBuf,
    pub package_root: Option<PathBuf>,
    pub source_type: PySourceType,
    pub python_version: PythonVersion,
    pub platform: String,
    pub typing_modules: Vec<String>,
    pub custom_builtins: Vec<String>,
    pub max_rows: usize,
    pub max_node_visits: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraversalStats {
    pub rows: usize,
    /// Includes callbacks attempted while a current traversal pass drains after cancellation.
    pub node_visits: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Cancelled,
    RowBudget,
    WorkBudget,
    Sink(String),
    InvalidSyntax,
    UnsupportedSyntax,
    UnsupportedSourceKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Incomplete {
    pub reason: StopReason,
    pub stats: TraversalStats,
}

/// Fallible owned row receiver. Checkpoints support cancellation even between emitted rows.
pub trait Sink {
    fn checkpoint(&mut self) -> Result<(), StopReason> {
        Ok(())
    }
    fn observe(&mut self, fact: Fact) -> Result<(), StopReason>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeOrigin {
    Source,
    /// Parsed inside an annotation string by the existing driver; not canonical source syntax.
    StringAnnotation,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub range: TextRange,
    pub kind: NodeKind,
    pub expr_context: Option<ruff_python_ast::ExprContext>,
    pub origin: NodeOrigin,
    pub scope: u32,
    pub definition: u32,
    pub flags: u32,
    pub context: ExecutionContext,
    /// Lexical name-load resolution at this phase; final references are authoritative.
    pub binding: Option<u32>,
    pub qualified_name: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Scope {
    pub id: u32,
    pub parent: Option<u32>,
    pub kind: &'static str,
    pub range: Option<TextRange>,
}

#[derive(Debug, Clone)]
pub struct Binding {
    pub id: u32,
    pub name: String,
    pub range: TextRange,
    pub scope: u32,
    pub kind: &'static str,
    pub context: ExecutionContext,
    pub flags: u16,
    pub qualified_name: Option<Vec<String>>,
    pub shadowed: Option<u32>,
    pub outer_shadowed: Option<u32>,
    pub definition_scope: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Reference {
    pub id: u32,
    pub binding: u32,
    pub scope: u32,
    pub range: TextRange,
    pub is_load: bool,
    pub typing_context: bool,
    pub typing_only_annotation: bool,
    pub runtime_annotation: bool,
    pub string_annotation: bool,
    pub type_checking: bool,
}

#[derive(Debug, Clone)]
pub enum Fact {
    Input(Settings),
    Node(Node),
    Scope(Scope),
    Definition {
        id: u32,
        parent: Option<u32>,
        name: Option<String>,
        range: Option<TextRange>,
        kind: &'static str,
    },
    Binding(Binding),
    Reference(Reference),
    Unresolved {
        range: TextRange,
        name: String,
        wildcard_import: bool,
        annotation_binding: Option<u32>,
    },
}

/// Parse and observe a Python module with no diagnostic rules or fixes. A successful return only
/// certifies completion of Ruff's semantic passes, not completeness of Python runtime semantics.
/// Notebook cells require Ruff's notebook orchestration and are explicitly unsupported here.
pub fn semantic_facts(
    source: &str,
    settings: &Settings,
    sink: &mut dyn Sink,
) -> Result<TraversalStats, Incomplete> {
    // Check before the upstream parser's indivisible call as well as during observation.
    sink.checkpoint().map_err(|reason| Incomplete {
        reason,
        stats: TraversalStats::default(),
    })?;
    if settings.source_type.is_ipynb() {
        return Err(Incomplete {
            reason: StopReason::UnsupportedSourceKind,
            stats: TraversalStats::default(),
        });
    }
    if settings.max_rows == 0 {
        return Err(Incomplete {
            reason: StopReason::RowBudget,
            stats: TraversalStats::default(),
        });
    }
    let parsed = parse_unchecked(
        source,
        ParseOptions::from(settings.source_type).with_target_version(settings.python_version),
    )
    .try_into_module()
    .expect("Python source types parse as modules");
    observe_parsed(source, &parsed, settings, sink)
}

/// Observe an already parsed module using the same driver and validation as [`semantic_facts`].
/// This lets callers reuse the latest Ruff parse for a canonical syntax walk and lexical facts.
/// `source` must be the exact text that produced `parsed`, parsed with the supplied source kind
/// and Python version. The caller owns this snapshot agreement: `Parsed` does not retain text or
/// parse options. Node ranges and parsed string annotations rely on that precondition.
pub fn observe_parsed(
    source: &str,
    parsed: &Parsed<ruff_python_ast::ModModule>,
    settings: &Settings,
    sink: &mut dyn Sink,
) -> Result<TraversalStats, Incomplete> {
    let observation = validated_observation(parsed, settings, sink)?;
    let preparation = SourcePreparation::new(source, parsed);
    run(parsed, source, settings, observation, &preparation)
}

/// Source-derived immutable preparation. The source lifetime is independent of the parsed
/// input's borrow; token indexes are owned and style/location only borrow the source text.
/// The caller supplies the exact source that produced `parsed`.
pub struct SourcePreparation<'s> {
    locator: Locator<'s>,
    stylist: Stylist<'s>,
    indexer: Indexer,
}
impl<'s> SourcePreparation<'s> {
    pub fn new(source: &'s str, parsed: &Parsed<ruff_python_ast::ModModule>) -> Self {
        Self {
            locator: Locator::new(source),
            stylist: Stylist::from_tokens(parsed.tokens(), source),
            indexer: Indexer::from_tokens(parsed.tokens(), source),
        }
    }
    pub fn locator(&self) -> &Locator<'s> {
        &self.locator
    }
    pub fn stylist(&self) -> &Stylist<'s> {
        &self.stylist
    }
    pub fn indexer(&self) -> &Indexer {
        &self.indexer
    }
}

/// Observe with source preparation shared by another native consumer. Source, parse options,
/// parsed tokens and preparation must correspond exactly, as for `observe_parsed`.
/// This preserves the empty-rule observation and its stopping/completion semantics.
pub fn observe_prepared(
    source: &str,
    parsed: &Parsed<ruff_python_ast::ModModule>,
    settings: &Settings,
    preparation: &SourcePreparation<'_>,
    sink: &mut dyn Sink,
) -> Result<TraversalStats, Incomplete> {
    let observation = validated_observation(parsed, settings, sink)?;
    run(parsed, source, settings, observation, preparation)
}

fn validated_observation<'s>(
    parsed: &Parsed<ruff_python_ast::ModModule>,
    settings: &Settings,
    sink: &'s mut dyn Sink,
) -> Result<Observation<'s>, Incomplete> {
    if settings.source_type.is_ipynb() {
        return Err(Incomplete {
            reason: StopReason::UnsupportedSourceKind,
            stats: TraversalStats::default(),
        });
    }
    let mut observation = Observation::new(sink, settings.max_rows, settings.max_node_visits);
    observation.emit(Fact::Input(settings.clone()));
    if observation.is_active() {
        if !parsed.errors().is_empty() {
            observation.stop(StopReason::InvalidSyntax);
        } else if !parsed.unsupported_syntax_errors().is_empty() {
            observation.stop(StopReason::UnsupportedSyntax);
        }
    }
    if observation.is_active() {
        Ok(observation)
    } else {
        Err(observation
            .into_result()
            .expect_err("inactive observation has a stop reason"))
    }
}

fn run(
    parsed: &Parsed<ruff_python_ast::ModModule>,
    source: &str,
    settings: &Settings,
    observation: Observation<'_>,
    preparation: &SourcePreparation<'_>,
) -> Result<TraversalStats, Incomplete> {
    let project_root = settings.package_root.clone().unwrap_or_else(|| {
        settings
            .path
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
    });
    let mut lint = LinterSettings::new_with_src(&project_root, vec![project_root.clone()]);
    lint.rules = crate::settings::rule_table::RuleTable::empty();
    lint.builtins.clone_from(&settings.custom_builtins);
    lint.typing_modules.clone_from(&settings.typing_modules);
    lint.unresolved_target_version = TargetVersion(Some(settings.python_version));
    let locator = preparation.locator();
    let stylist = preparation.stylist();
    let indexer = preparation.indexer();
    let noqa = NoqaMapping::default();
    let context = LintContext::new(&settings.path, source, &lint);
    let (semantic_errors, result) = check_ast_with_observation(
        parsed,
        locator,
        stylist,
        indexer,
        &noqa,
        &lint,
        flags::Noqa::Disabled,
        &settings.path,
        settings.package_root.as_deref().map(PackageRoot::root),
        settings.source_type,
        None,
        None,
        lint.unresolved_target_version,
        &context,
        Some(observation),
    );
    match result {
        Ok(stats) if !semantic_errors.is_empty() => Err(Incomplete {
            reason: StopReason::InvalidSyntax,
            stats,
        }),
        result => result,
    }
}

pub(crate) struct Observation<'a> {
    sink: &'a mut dyn Sink,
    max_rows: usize,
    max_node_visits: usize,
    stats: TraversalStats,
    reason: Option<StopReason>,
}

impl<'a> Observation<'a> {
    fn new(sink: &'a mut dyn Sink, max_rows: usize, max_node_visits: usize) -> Self {
        Self {
            sink,
            max_rows,
            max_node_visits,
            stats: TraversalStats::default(),
            reason: None,
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        self.reason.is_none()
    }

    fn stop(&mut self, reason: StopReason) {
        if self.reason.is_none() {
            self.reason = Some(reason);
        }
    }

    pub(crate) fn checkpoint(&mut self) -> bool {
        self.stats.node_visits = self.stats.node_visits.saturating_add(1);
        if !self.is_active() {
            return false;
        }
        if self.stats.node_visits > self.max_node_visits {
            self.stop(StopReason::WorkBudget);
            return false;
        }
        if let Err(reason) = self.sink.checkpoint() {
            self.stop(reason);
        }
        self.is_active()
    }

    fn emit(&mut self, fact: Fact) {
        if !self.is_active() {
            return;
        }
        if self.stats.rows >= self.max_rows {
            self.stop(StopReason::RowBudget);
            return;
        }
        if let Err(reason) = self.sink.checkpoint() {
            self.stop(reason);
            return;
        }
        match self.sink.observe(fact) {
            Ok(()) => self.stats.rows += 1,
            Err(reason) => self.stop(reason),
        }
    }

    pub(crate) fn into_result(self) -> Result<TraversalStats, Incomplete> {
        match self.reason {
            Some(reason) => Err(Incomplete {
                reason,
                stats: self.stats,
            }),
            None => Ok(self.stats),
        }
    }

    pub(crate) fn node(
        &mut self,
        node: AnyNodeRef<'_>,
        semantic: &SemanticModel<'_>,
        expr: Option<&Expr>,
    ) {
        if !self.is_active() {
            return;
        }
        self.emit(Fact::Node(Node {
            range: node.range(),
            kind: node.kind(),
            expr_context: expr.and_then(Expr::as_name_expr).map(|name| name.ctx),
            scope: semantic.scope_id.as_u32(),
            definition: semantic.definition_id.as_u32(),
            origin: if semantic.in_string_type_definition() {
                NodeOrigin::StringAnnotation
            } else {
                NodeOrigin::Source
            },
            flags: semantic.flags.bits(),
            context: semantic.execution_context(),
            binding: expr
                .and_then(Expr::as_name_expr)
                .and_then(|name| semantic.resolve_name(name))
                .map(|id| id.as_u32()),
            qualified_name: expr
                .and_then(|expr| semantic.resolve_qualified_name(expr))
                .map(|name| {
                    name.segments()
                        .iter()
                        .map(|segment| (*segment).to_owned())
                        .collect()
                }),
        }));
    }

    pub(crate) fn finish(&mut self, semantic: &SemanticModel<'_>, source: &str) {
        if !self.is_active() {
            return;
        }
        for (id, scope) in semantic.scopes.iter_enumerated() {
            if !self.is_active() {
                return;
            }
            let (kind, range) = match scope.kind {
                ScopeKind::Module => ("Module", None),
                ScopeKind::Class(node) => ("Class", Some(node.range())),
                ScopeKind::Function(node) => ("Function", Some(node.range())),
                ScopeKind::Lambda(node) => ("Lambda", Some(node.range())),
                ScopeKind::Generator { .. } => ("Generator", None),
                ScopeKind::Type => ("Type", None),
                ScopeKind::DunderClassCell => ("DunderClassCell", None),
            };
            self.emit(Fact::Scope(Scope {
                id: id.as_u32(),
                parent: semantic
                    .scopes
                    .ancestor_ids(id)
                    .nth(1)
                    .map(|id| id.as_u32()),
                kind,
                range,
            }));
        }
        for (id, definition) in semantic.definitions.iter_enumerated() {
            if !self.is_active() {
                return;
            }
            let (parent, range, kind) = match definition {
                Definition::Module(_) => (None, None, "Module"),
                Definition::Member(member) => (
                    Some(member.parent.as_u32()),
                    Some(member.range()),
                    match member.kind {
                        ruff_python_semantic::MemberKind::Class(_) => "Class",
                        ruff_python_semantic::MemberKind::NestedClass(_) => "NestedClass",
                        ruff_python_semantic::MemberKind::Function(_) => "Function",
                        ruff_python_semantic::MemberKind::NestedFunction(_) => "NestedFunction",
                        ruff_python_semantic::MemberKind::Method(_) => "Method",
                    },
                ),
            };
            self.emit(Fact::Definition {
                id: id.as_u32(),
                parent,
                name: definition.name().map(str::to_owned),
                range,
                kind,
            });
        }
        for (id, binding) in semantic.bindings.iter_enumerated() {
            if !self.is_active() {
                return;
            }
            let kind = binding_kind(&binding.kind);
            self.emit(Fact::Binding(Binding {
                id: id.as_u32(),
                name: if binding.kind.is_builtin() {
                    semantic.scopes[binding.scope]
                        .all_bindings()
                        .find_map(|(name, candidate)| (candidate == id).then(|| name.to_owned()))
                        .unwrap_or_default()
                } else {
                    binding.name(source).to_owned()
                },
                range: binding.range,
                scope: binding.scope.as_u32(),
                kind,
                context: binding.context,
                flags: binding.flags.bits(),
                qualified_name: binding.as_any_import().map(|import| {
                    import
                        .qualified_name()
                        .segments()
                        .iter()
                        .map(|segment| (*segment).to_owned())
                        .collect()
                }),
                shadowed: semantic.scopes[binding.scope]
                    .shadowed_binding(id)
                    .map(|id| id.as_u32()),
                outer_shadowed: semantic.shadowed_bindings.get(&id).map(|id| id.as_u32()),
                definition_scope: match binding.kind {
                    BindingKind::ClassDefinition(scope)
                    | BindingKind::FunctionDefinition(scope) => Some(scope.as_u32()),
                    _ => None,
                },
            }));
            for reference_id in binding.references() {
                if !self.is_active() {
                    return;
                }
                let reference = semantic.reference(reference_id);
                self.emit(Fact::Reference(Reference {
                    id: reference_id.as_u32(),
                    binding: id.as_u32(),
                    scope: reference.scope_id().as_u32(),
                    range: reference.range(),
                    is_load: reference.is_load(),
                    typing_context: reference.in_typing_context(),
                    typing_only_annotation: reference.in_typing_only_annotation(),
                    runtime_annotation: reference.in_runtime_evaluated_annotation(),
                    string_annotation: reference.in_string_type_definition(),
                    type_checking: reference.in_type_checking_block(),
                }));
            }
        }
        for reference in semantic.unresolved_references() {
            if !self.is_active() {
                return;
            }
            self.emit(Fact::Unresolved {
                range: reference.range(),
                name: reference.name(source).to_owned(),
                wildcard_import: reference.is_wildcard_import(),
                annotation_binding: reference.annotation_binding_id().map(|id| id.as_u32()),
            });
        }
    }
}

fn binding_kind(kind: &BindingKind<'_>) -> &'static str {
    match kind {
        BindingKind::Annotation => "Annotation",
        BindingKind::Argument => "Argument",
        BindingKind::NamedExprAssignment => "NamedExprAssignment",
        BindingKind::Assignment => "Assignment",
        BindingKind::TypeParam => "TypeParam",
        BindingKind::LoopVar => "LoopVar",
        BindingKind::WithItemVar => "WithItemVar",
        BindingKind::Global(_) => "Global",
        BindingKind::Nonlocal(_, _) => "Nonlocal",
        BindingKind::Builtin => "Builtin",
        BindingKind::ClassDefinition(_) => "ClassDefinition",
        BindingKind::FunctionDefinition(_) => "FunctionDefinition",
        BindingKind::Export(_) => "Export",
        BindingKind::FutureImport => "FutureImport",
        BindingKind::Import(_) => "Import",
        BindingKind::FromImport(_) => "FromImport",
        BindingKind::SubmoduleImport(_) => "SubmoduleImport",
        BindingKind::Deletion => "Deletion",
        BindingKind::BoundException => "BoundException",
        BindingKind::UnboundException(_) => "UnboundException",
        BindingKind::DunderClassCell => "DunderClassCell",
    }
}

// Erase the sink lifetime before lending the observer to the locally allocated Checker.
// RefCell only mediates synchronous callbacks; no model or AST borrow escapes a callback.
pub(crate) trait Observer {
    fn is_active(&self) -> bool;
    fn checkpoint(&self) -> bool;
    fn node(&self, node: AnyNodeRef<'_>, semantic: &SemanticModel<'_>, expr: Option<&Expr>);
    fn finish(&self, semantic: &SemanticModel<'_>, source: &str);
}

impl Observer for std::cell::RefCell<Observation<'_>> {
    fn is_active(&self) -> bool {
        self.borrow().is_active()
    }
    fn checkpoint(&self) -> bool {
        self.borrow_mut().checkpoint()
    }
    fn node(&self, node: AnyNodeRef<'_>, semantic: &SemanticModel<'_>, expr: Option<&Expr>) {
        self.borrow_mut().node(node, semantic, expr);
    }
    fn finish(&self, semantic: &SemanticModel<'_>, source: &str) {
        self.borrow_mut().finish(semantic, source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Rule;

    #[test]
    fn shared_source_preparation_preserves_standalone_rows_and_stop_statistics() {
        #[derive(Default)]
        struct Rows {
            rows: Vec<String>,
            checks: usize,
            stop: Option<StopReason>,
        }
        impl Sink for Rows {
            fn checkpoint(&mut self) -> Result<(), StopReason> {
                self.checks += 1;
                if self.checks >= 5
                    && let Some(reason) = &self.stop
                {
                    return Err(reason.clone());
                }
                Ok(())
            }
            fn observe(&mut self, fact: Fact) -> Result<(), StopReason> {
                self.rows.push(format!("{fact:?}"));
                Ok(())
            }
        }
        for source in [
            "import os\nmissing\ndef f():\n    unused = 1\n    return absent\n",
            "from os import (\n    path,  # noqa: F401\n    environ,\n)\n",
            "from __future__ import annotations\nfrom typing import TYPE_CHECKING as TC\nif TC:\n    import json\ndef f(x: 'Later'):\n    return lambda: missing\nclass Later:\n    pass\n",
            "def broken(:\n    missing\n",
            "type Alias = int\n",
        ] {
            for (rows, visits, stop) in [
                (10000, 10000, None),
                (0, 10000, None),
                (3, 10000, None),
                (10000, 1, None),
                (10000, 6, None),
                (10000, 10000, Some(StopReason::Cancelled)),
                (
                    10000,
                    10000,
                    Some(StopReason::Sink("resource refusal".into())),
                ),
            ] {
                let settings = Settings {
                    path: PathBuf::from("/explicit/module.py"),
                    package_root: Some(PathBuf::from("/explicit")),
                    source_type: PySourceType::Python,
                    python_version: PythonVersion::PY311,
                    platform: "linux".into(),
                    typing_modules: vec!["custom_typing".into()],
                    custom_builtins: vec!["provided".into()],
                    max_rows: rows,
                    max_node_visits: visits,
                };
                let parsed = parse_unchecked(
                    source,
                    ParseOptions::from(settings.source_type)
                        .with_target_version(settings.python_version),
                )
                .try_into_module()
                .unwrap();
                let mut standalone = Rows {
                    stop: stop.clone(),
                    ..Rows::default()
                };
                let expected = observe_parsed(source, &parsed, &settings, &mut standalone);
                let preparation = SourcePreparation::new(source, &parsed);
                let mut shared = Rows {
                    stop,
                    ..Rows::default()
                };
                let actual =
                    observe_prepared(source, &parsed, &settings, &preparation, &mut shared);
                assert_eq!(
                    actual, expected,
                    "source={source:?} rows={rows} visits={visits}"
                );
                assert_eq!(shared.rows, standalone.rows);
                assert_eq!(shared.checks, standalone.checks);
                assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
            }
        }
    }

    #[test]
    fn no_observer_lint_parity() {
        struct Rows(Vec<Fact>);
        impl Sink for Rows {
            fn observe(&mut self, fact: Fact) -> Result<(), StopReason> {
                self.0.push(fact);
                Ok(())
            }
        }
        for source in [
            "import os\nmissing\n",
            "from __future__ import annotations\nfrom typing import TYPE_CHECKING as TC\nif TC:\n    import json\ndef f(x: Later):\n    return unresolved\nclass Later:\n    pass\n",
        ] {
            let path = Path::new("/explicit/module.py");
            let parsed = ruff_python_parser::parse_module(source).unwrap();
            let settings = LinterSettings::for_rules([Rule::UnusedImport, Rule::UndefinedName]);
            let locator = Locator::new(source);
            let stylist = Stylist::from_tokens(parsed.tokens(), source);
            let indexer = Indexer::from_tokens(parsed.tokens(), source);
            let noqa = NoqaMapping::default();
            let context = LintContext::new(path, source, &settings);
            let plain_errors = crate::checkers::ast::check_ast(
                &parsed,
                &locator,
                &stylist,
                &indexer,
                &noqa,
                &settings,
                flags::Noqa::Disabled,
                path,
                None,
                PySourceType::Python,
                None,
                None,
                settings.unresolved_target_version,
                &context,
            );
            let plain = context.into_diagnostics();
            let context = LintContext::new(path, source, &settings);
            let mut rows = Rows(vec![]);
            let (observed_errors, result) = check_ast_with_observation(
                &parsed,
                &locator,
                &stylist,
                &indexer,
                &noqa,
                &settings,
                flags::Noqa::Disabled,
                path,
                None,
                PySourceType::Python,
                None,
                None,
                settings.unresolved_target_version,
                &context,
                Some(Observation::new(&mut rows, 10000, 10000)),
            );
            result.unwrap();
            let observed = context.into_diagnostics();
            assert!(!plain.is_empty());
            assert!(!rows.0.is_empty());
            assert_eq!(format!("{plain:?}"), format!("{observed:?}"));
            assert_eq!(format!("{plain_errors:?}"), format!("{observed_errors:?}"));
        }
    }
}
