import { useLayoutEffect, useRef } from 'react';
import { EditorState, StateEffect, StateField } from '@codemirror/state';
import { Decoration, EditorView, highlightSpecialChars, keymap, lineNumbers } from '@codemirror/view';
import { closeSearchPanel, openSearchPanel, search, searchKeymap } from '@codemirror/search';
import { HighlightStyle, StreamLanguage, syntaxHighlighting } from '@codemirror/language';
import { tags } from '@lezer/highlight';
import { python } from '@codemirror/lang-python';
import type { Stage } from './core/locations.ts';

export type Mark = {line: number; quality: 'metadata' | 'approximate' | 'selected'};
const setMarks = StateEffect.define<Mark[]>();
const marksField = StateField.define({
  create: () => Decoration.none,
  update(value, tx) {
    for (const effect of tx.effects) if (effect.is(setMarks)) {
      return Decoration.set(effect.value.filter(m => m.line > 0 && m.line <= tx.state.doc.lines).map(m =>
        Decoration.line({class: `correlation-${m.quality}`}).range(tx.state.doc.line(m.line).from)), true);
    }
    return value.map(tx.changes);
  },
  provide: field => EditorView.decorations.from(field),
});
const style = HighlightStyle.define([
  {tag: [tags.comment, tags.lineComment, tags.blockComment], color: 'var(--syntax-comment)'},
  {tag: [tags.keyword, tags.controlKeyword, tags.operatorKeyword], color: 'var(--syntax-keyword)'},
  {tag: [tags.number, tags.bool, tags.null], color: 'var(--syntax-number)'},
  {tag: [tags.typeName, tags.className, tags.atom], color: 'var(--syntax-type)'},
  {tag: [tags.string, tags.special(tags.string)], color: 'var(--syntax-string)'},
  {tag: [tags.variableName, tags.labelName], color: 'var(--syntax-var)'},
  {tag: tags.definition(tags.variableName), color: 'var(--syntax-keyword)'},
]);
const lowLevel = StreamLanguage.define<{comment: boolean}>({
  name: 'GPU compiler text', startState: () => ({comment:false}),
  token(stream, state) {
    if (state.comment) {
      if (stream.skipTo('*/')) {stream.match('*/'); state.comment = false;}
      else stream.skipToEnd();
      return 'comment';
    }
    if (stream.eatSpace()) return null;
    if (stream.match('//')) {stream.skipToEnd(); return 'comment';}
    if (stream.match('/*')) {state.comment = true; return 'comment';}
    if (stream.match(/^"(?:[^"\\]|\\.)*"/)) return 'string';
    if (stream.match(/^[%@!#][\w.$-]+/)) return 'variableName';
    if (stream.match(/^(?:0x[\da-fA-F]+|[-+]?\d+(?:\.\d+)?)(?![\w])/)) return 'number';
    if (stream.match(/^(?:[.][\w.$-]+|(?:arith|cute|cute_nvgpu|func|gpu|llvm|memref|nvvm|scf|vector)\.[\w.]+|[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*\b|(?:ld|st|mov|add|sub|mul|mad|fma|cvt|bar|fence|ret|bra|setp)(?:\.[\w]+)*\b)/)) return 'keyword';
    if (stream.match(/^(?:f16|f32|f64|i1|i32|i64|index|vector|tensor|memref)\b/)) return 'typeName';
    if (stream.match(/^[\w.$-]+/)) return null;
    stream.next(); return null;
  },
});

export function CodePane({stage, text, marks, targetLine, onLine, searchHandle}: {
  stage: Stage; text: string; marks: Mark[]; targetLine?: number; onLine: (n: number) => void;
  searchHandle: React.RefObject<(() => void) | null>;
}) {
  const element = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const callback = useRef(onLine);
  callback.current = onLine;
  useLayoutEffect(() => {
    const editor = new EditorView({parent:element.current!, state:EditorState.create({doc:text, extensions:[
      lineNumbers({domEventHandlers:{click(v, block) {callback.current(v.state.doc.lineAt(block.from).number); return false;}}}),
      EditorView.editable.of(false), EditorState.readOnly.of(true),
      EditorView.contentAttributes.of({tabindex:'0','aria-label': `${stage} code`}), highlightSpecialChars(),
      stage === 'python' ? python() : lowLevel, syntaxHighlighting(style),
      marksField, search({top:true}), keymap.of(searchKeymap),
      EditorView.domEventHandlers({
        click(e, v) {
          const row = e.target instanceof Element ? e.target.closest('.cm-line') : null;
          if (row) callback.current(v.state.doc.lineAt(v.posAtDOM(row)).number);
          return false;
        },
        keydown(e, v) {
          if (e.key === 'Enter' && e.target instanceof Element && e.target.closest('.cm-content')) {
            callback.current(v.state.doc.lineAt(v.state.selection.main.head).number); return true;
          }
          return false;
        },
      }),
    ]})});
    view.current = editor;
    searchHandle.current = () => {
      if (editor.dom.querySelector('.cm-search')) closeSearchPanel(editor);
      else openSearchPanel(editor);
    };
    return () => {searchHandle.current = null; view.current = null; editor.destroy();};
  }, [stage, text, searchHandle]);
  useLayoutEffect(() => {
    const v = view.current;
    if (!v) return;
    const effects: StateEffect<unknown>[] = [setMarks.of(marks)];
    if (targetLine && targetLine <= v.state.doc.lines) effects.push(EditorView.scrollIntoView(v.state.doc.line(targetLine).from, {y:'nearest',x:'nearest'}));
    v.dispatch({effects});
  }, [marks, targetLine, text]);
  return <div className="code-host" ref={element} />;
}
