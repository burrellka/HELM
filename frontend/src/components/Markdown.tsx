/**
 * A compact, safe Markdown renderer for advisor output.
 *
 * The advisor emits a consistent subset — headings, bold, inline code, bullet
 * and numbered lists, links, GFM tables, fenced code blocks, paragraphs — so a
 * focused renderer covers it without pulling a full markdown library into a
 * deliberately dependency-light frontend. It is SAFE by construction: it builds
 * React elements, never uses dangerouslySetInnerHTML, so model or web-search
 * text cannot inject markup.
 */
import { useMemo } from 'react';

type Align = 'left' | 'center' | 'right';
type Block =
  | { kind: 'h'; level: number; text: string }
  | { kind: 'p'; text: string }
  | { kind: 'ul'; items: string[] }
  | { kind: 'ol'; items: string[] }
  | { kind: 'code'; text: string }
  | { kind: 'table'; header: string[]; align: Align[]; rows: string[][] };

/** Split a `| a | b |` row into trimmed cells, dropping the empty edges that
 *  leading/trailing pipes produce. */
function splitRow(line: string): string[] {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|')) s = s.slice(0, -1);
  return s.split('|').map((c) => c.trim());
}

/** True for a GFM separator row: every cell is dashes with optional edge colons
 *  (`---`, `:--`, `--:`, `:-:`). This is what distinguishes a table from a
 *  paragraph that merely contains pipes. */
function isSeparatorRow(line: string): boolean {
  if (!line.includes('|') && !line.includes('-')) return false;
  const cells = splitRow(line);
  return cells.length > 0 && cells.every((c) => /^:?-{1,}:?$/.test(c));
}

function alignOf(cell: string): Align {
  const l = cell.startsWith(':'), r = cell.endsWith(':');
  return l && r ? 'center' : r ? 'right' : 'left';
}

function looksLikeRow(line: string): boolean {
  return line.includes('|');
}

function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  const blocks: Block[] = [];
  let para: string[] = [];

  const flushPara = () => {
    if (para.length) { blocks.push({ kind: 'p', text: para.join(' ') }); para = []; }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
    const numbered = /^\s*\d+\.\s+(.*)$/.exec(line);
    const fence = /^\s*```(.*)$/.exec(line);

    // Fenced code block: consume verbatim until the closing ```.
    if (fence) {
      flushPara();
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) { body.push(lines[i]); i++; }
      blocks.push({ kind: 'code', text: body.join('\n') });
      continue;
    }

    // GFM table: a row line immediately followed by a separator row. Parse the
    // header, the alignment, then every following row-shaped line.
    if (looksLikeRow(line) && i + 1 < lines.length && isSeparatorRow(lines[i + 1])) {
      flushPara();
      const header = splitRow(line);
      const align = splitRow(lines[i + 1]).map(alignOf);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && looksLikeRow(lines[i]) && lines[i].trim() !== '') {
        rows.push(splitRow(lines[i]));
        i++;
      }
      i--; // the for-loop will ++ back to the first non-row line
      blocks.push({ kind: 'table', header, align, rows });
      continue;
    }

    if (heading) {
      flushPara();
      blocks.push({ kind: 'h', level: heading[1].length, text: heading[2] });
    } else if (bullet) {
      flushPara();
      const items = [bullet[1]];
      while (i + 1 < lines.length) {
        const next = /^\s*[-*]\s+(.*)$/.exec(lines[i + 1]);
        if (!next) break;
        items.push(next[1]); i++;
      }
      blocks.push({ kind: 'ul', items });
    } else if (numbered) {
      flushPara();
      const items = [numbered[1]];
      while (i + 1 < lines.length) {
        const next = /^\s*\d+\.\s+(.*)$/.exec(lines[i + 1]);
        if (!next) break;
        items.push(next[1]); i++;
      }
      blocks.push({ kind: 'ol', items });
    } else if (line.trim() === '') {
      flushPara();
    } else {
      para.push(line.trim());
    }
  }
  flushPara();
  return blocks;
}

/** Inline formatting: **bold**, `code`, [text](url), *italic*. */
function inline(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const re = /(\*\*([^*]+)\*\*)|(`([^`]+)`)|(\[([^\]]+)\]\(([^)]+)\))|(\*([^*\n]+)\*)/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[1]) nodes.push(<strong key={key++}>{m[2]}</strong>);
    else if (m[3]) nodes.push(<code key={key++} className="mono">{m[4]}</code>);
    else if (m[5]) {
      const href = m[7];
      const safe = /^https?:\/\//i.test(href);  // only http(s) links become anchors
      nodes.push(safe
        ? <a key={key++} href={href} target="_blank" rel="noreferrer noopener"
             style={{ color: 'var(--accent-primary)' }}>{m[6]}</a>
        : <span key={key++}>{m[6]}</span>);
    } else if (m[8]) nodes.push(<em key={key++}>{m[9]}</em>);
    last = re.lastIndex;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

const H_SIZE: Record<number, string> = {
  1: '1.15rem', 2: '1.1rem', 3: '1rem', 4: '0.9375rem', 5: '0.875rem', 6: '0.875rem',
};

export default function Markdown({ children }: { children: string }) {
  const blocks = useMemo(() => parseBlocks(children || ''), [children]);
  return (
    <div style={{ fontSize: '0.9rem', lineHeight: 1.6 }}>
      {blocks.map((b, i) => {
        if (b.kind === 'h') {
          return (
            <div key={i} style={{
              fontSize: H_SIZE[b.level] ?? '1rem', fontWeight: 700,
              margin: i === 0 ? '0 0 0.4rem' : '0.9rem 0 0.4rem',
              color: 'var(--text-primary)',
            }}>{inline(b.text)}</div>
          );
        }
        if (b.kind === 'ul') {
          return (
            <ul key={i} style={{ margin: '0.35rem 0', paddingLeft: '1.25rem' }}>
              {b.items.map((it, j) => (
                <li key={j} style={{ margin: '0.2rem 0' }}>{inline(it)}</li>
              ))}
            </ul>
          );
        }
        if (b.kind === 'ol') {
          return (
            <ol key={i} style={{ margin: '0.35rem 0', paddingLeft: '1.35rem' }}>
              {b.items.map((it, j) => (
                <li key={j} style={{ margin: '0.2rem 0' }}>{inline(it)}</li>
              ))}
            </ol>
          );
        }
        if (b.kind === 'code') {
          return (
            <pre key={i} style={{
              margin: '0.5rem 0', padding: '0.6rem 0.75rem', overflowX: 'auto',
              background: 'var(--bg-subtle)', border: '1px solid var(--border-color)',
              borderRadius: 8, fontSize: '0.8125rem', lineHeight: 1.45,
            }}>
              <code className="mono">{b.text}</code>
            </pre>
          );
        }
        if (b.kind === 'table') {
          // Wrapped so a wide table scrolls inside its own box instead of
          // forcing the whole chat bubble to scroll sideways.
          return (
            <div key={i} style={{ overflowX: 'auto', margin: '0.6rem 0' }}>
              <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '0.8125rem' }}>
                <thead>
                  <tr>
                    {b.header.map((h, j) => (
                      <th key={j} style={{
                        textAlign: b.align[j] ?? 'left', padding: '0.35rem 0.6rem',
                        borderBottom: '2px solid var(--border-strong)', fontWeight: 700,
                        whiteSpace: 'nowrap',
                      }}>{inline(h)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {b.rows.map((row, r) => (
                    <tr key={r}>
                      {row.map((cell, c) => (
                        <td key={c} style={{
                          textAlign: b.align[c] ?? 'left', padding: '0.35rem 0.6rem',
                          borderBottom: '1px solid var(--border-color)', verticalAlign: 'top',
                        }}>{inline(cell)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return <p key={i} style={{ margin: '0.5rem 0' }}>{inline(b.text)}</p>;
      })}
    </div>
  );
}
