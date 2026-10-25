import './style.css';
import {
  Backend, ColumnKind, Op, decodeServer, encodeClient,
  type ClientMessage, type Column, type PivotCell, type Predicate, type Schema, type SortKey, type Stats,
  type ViewSpec, type ViewUpdate,
} from './proto';

// ---------------------------------------------------------------- schema helpers

class Cols {
  readonly byName = new Map<string, Column>();
  constructor(readonly schema: Schema) {
    for (const c of schema.columns) {
      c.dict ??= [];
      this.byName.set(c.name, c);
    }
  }
  id(name: string): number {
    const c = this.byName.get(name);
    if (!c) throw new Error(`unknown column ${name}`);
    return c.id ?? 0;
  }
  col(id: number): Column { return this.schema.columns[id]; }
  cat(name: string, value: string): number {
    const i = this.byName.get(name)?.dict.indexOf(value) ?? -1;
    if (i < 0) throw new Error(`unknown ${name} ${value}`);
    return i;
  }
  mask(name: string, values: string[]): number {
    return values.reduce((m, v) => m | (1 << this.cat(name, v)), 0);
  }
}

// ---------------------------------------------------------------- formatting

const int = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
const px = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const bps = new Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const PRICE_COLS = new Set(['limit_px', 'avg_px', 'mid']);
const TS_COLS = new Set(['create_ts', 'update_ts']);

function fmtTime(ms: number): string {
  const h = Math.floor(ms / 3_600_000), m = Math.floor(ms / 60_000) % 60, s = Math.floor(ms / 1000) % 60;
  const p = (n: number, w = 2) => String(n).padStart(w, '0');
  return `${p(h)}:${p(m)}:${p(s)}.${p(ms % 1000, 3)}`;
}

function fmtCell(c: Column, v: number): string {
  if (c.kind === ColumnKind.CATEGORY) return c.dict[v] ?? String(v);
  if (TS_COLS.has(c.name)) return fmtTime(v);
  if (c.name === 'order_id') return String(v);
  if (PRICE_COLS.has(c.name)) return px.format(v);
  if (c.name === 'dist_bps') return bps.format(v);
  return int.format(v);
}

function cellClass(c: Column, v: number): string {
  if (c.name === 'side') return v === 0 ? 'buy' : 'sell';
  if (c.name === 'pnl') return v < 0 ? 'neg' : v > 0 ? 'pos' : '';
  return c.kind === ColumnKind.CATEGORY ? '' : 'num';
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = '', text = ''): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

// ---------------------------------------------------------------- panel definitions

interface PanelDef {
  title: string;
  filter: (c: Cols) => Predicate[];
  sort: [string, boolean][];  // column, desc
  columns: string[];
  limit: number;
  pivot?: [string, string];
}

const open = (c: Cols): Predicate => ({ column: c.id('status'), op: Op.IN, mask: c.mask('status', ['NEW', 'PARTIAL']) });

const PANELS: PanelDef[] = [
  {
    title: 'Largest open notional',
    filter: (c) => [open(c)],
    sort: [['notional', true]],
    columns: ['order_id', 'instrument', 'side', 'trader', 'status', 'qty', 'leaves', 'limit_px', 'mid', 'notional', 'dist_bps'],
    limit: 20,
  },
  {
    title: 'Closest to market (passive orders)',
    filter: (c) => [open(c), { column: c.id('dist_bps'), op: Op.GE, value: 0 }],
    sort: [['dist_bps', false]],
    columns: ['order_id', 'instrument', 'side', 'venue', 'strategy', 'limit_px', 'mid', 'dist_bps', 'leaves', 'notional'],
    limit: 20,
  },
  {
    title: 'Worst P&L on fills',
    filter: (c) => [{ column: c.id('filled'), op: Op.GT, value: 0 }],
    sort: [['pnl', false]],
    columns: ['order_id', 'instrument', 'side', 'desk', 'trader', 'filled', 'avg_px', 'mid', 'pnl'],
    limit: 20,
  },
  {
    title: 'Latest activity',
    filter: () => [],
    sort: [['update_ts', true]],
    columns: ['update_ts', 'order_id', 'instrument', 'side', 'status', 'qty', 'filled', 'avg_px', 'venue'],
    limit: 20,
  },
  {
    title: 'Pivot: desk × status (all orders)',
    filter: () => [],
    sort: [],
    columns: [],
    limit: 0,
    pivot: ['desk', 'status'],
  },
  {
    title: 'Pivot: strategy × venue (open orders)',
    filter: (c) => [open(c)],
    sort: [],
    columns: [],
    limit: 0,
    pivot: ['strategy', 'venue'],
  },
];

const MEASURES: [keyof PivotCell, string][] = [
  ['sum_notional', 'notional'], ['sum_leaves', 'leaves'], ['sum_pnl', 'P&L'], ['count', 'orders'],
];

// ---------------------------------------------------------------- panel

class Panel {
  readonly root = el('section', 'panel');
  private readonly info = el('span', 'info');
  private readonly badge = el('span', 'badge');
  private readonly body = el('div', 'body');
  private spec: ViewSpec;
  private total = 0;
  private measure: keyof PivotCell = 'sum_notional';
  private last = new Map<number, number[]>();  // row -> values, for change flashes
  private lastUpdate: ViewUpdate | null = null;

  constructor(readonly id: number, private readonly def: PanelDef, private readonly cols: Cols,
              private readonly send: (m: ClientMessage) => void) {
    this.spec = {
      filter: def.filter(cols),
      sort: def.sort.map(([n, desc]) => ({ column: cols.id(n), desc })),
      offset: 0,
      limit: def.limit,
      pivot_by: def.pivot?.map((n) => cols.id(n)) ?? [],
    };
    const head = el('header');
    head.append(el('h2', '', def.title));
    const desc = el('div', 'desc', this.describe());
    if (def.pivot) {
      const sel = el('select');
      for (const [k, label] of MEASURES) {
        const o = el('option', '', label);
        o.value = k;
        sel.append(o);
      }
      sel.onchange = () => {
        this.measure = sel.value as keyof PivotCell;
        if (this.lastUpdate) this.render(this.lastUpdate);
      };
      head.append(sel);
    } else {
      const nav = el('div', 'nav');
      const btn = (t: string, f: () => number) => {
        const b = el('button', '', t);
        b.onclick = () => this.scrollTo(f());
        nav.append(b);
      };
      btn('⏮', () => 0);
      btn('◀', () => (this.spec.offset ?? 0) - this.def.limit);
      btn('▶', () => (this.spec.offset ?? 0) + this.def.limit);
      btn('⏭', () => this.total - this.def.limit);
      head.append(nav);
      this.body.addEventListener('wheel', (e) => {
        e.preventDefault();
        this.scrollTo((this.spec.offset ?? 0) + Math.sign(e.deltaY) * 3);
      }, { passive: false });
    }
    const status = el('div', 'status');
    status.append(this.info, this.badge);
    this.root.append(head, desc, status, this.body);
  }

  subscribe(): void { this.send({ subscribe: { view_id: this.id, spec: this.spec } }); }

  private scrollTo(offset: number): void {
    const max = Math.max(0, this.total - this.def.limit);
    const o = Math.max(0, Math.min(max, Math.round(offset)));
    if (o === this.spec.offset) return;
    this.spec.offset = o;
    this.subscribe();
  }

  private describe(): string {
    const c = this.cols;
    const ops = ['=', '≠', '<', '≤', '>', '≥'];
    const f = (this.spec.filter ?? []).map((p) => {
      const col = c.col(p.column);
      if (p.op === Op.IN) return `${col.name} in (${col.dict.filter((_, i) => (p.mask ?? 0) >> i & 1).join(', ')})`;
      const v = col.kind === ColumnKind.CATEGORY ? col.dict[p.value ?? 0] : p.value ?? 0;
      return `${col.name} ${ops[p.op]} ${v}`;
    });
    const s = (this.spec.sort ?? []).map((k) => `${c.col(k.column).name} ${k.desc ? '↓' : '↑'}`);
    const parts = [`filter: ${f.length ? f.join(' and ') : 'none'}`];
    if (s.length) parts.push(`sort: ${s.join(', ')}`);
    if (this.def.pivot) parts.push(`pivot: ${this.def.pivot.join(' × ')}`);
    return parts.join('  ·  ');
  }

  private setSort(column: number, add: boolean): void {
    const sort = [...(this.spec.sort ?? [])];
    const i = sort.findIndex((k) => k.column === column);
    if (add) {
      if (i >= 0) sort[i] = { column, desc: !sort[i].desc };
      else if (sort.length < 3) sort.push({ column, desc: true });
    } else if (i === 0) {
      sort.splice(0, sort.length, { column, desc: !sort[0].desc });
    } else {
      sort.splice(0, sort.length, { column, desc: true });
    }
    this.spec.sort = sort;
    this.spec.offset = 0;
    (this.root.querySelector('.desc') as HTMLElement).textContent = this.describe();
    this.subscribe();
  }

  render(u: ViewUpdate): void {
    this.lastUpdate = u;
    this.total = u.total ?? 0;
    const off = u.offset ?? 0;
    if (this.def.pivot) {
      this.info.textContent = `${int.format(this.total)} orders in view`;
      this.renderPivot(u.pivot ?? []);
      return;
    }
    const n = u.rows?.length ?? 0;
    this.info.textContent = n
      ? `rows ${int.format(off + 1)}–${int.format(off + n)} of ${int.format(this.total)}`
      : `0 of ${int.format(this.total)} rows`;
    this.renderTable(u);
  }

  // Client-side check that the server's order is right on the primary sort column.
  private checkOrder(u: ViewUpdate): boolean {
    const k: SortKey | undefined = this.spec.sort?.[0];
    if (!k) return true;
    const rows = u.rows ?? [];
    for (let i = 1; i < rows.length; i++) {
      const a = rows[i - 1].values[k.column], b = rows[i].values[k.column];
      if (k.desc ? a < b : a > b) return false;
    }
    return true;
  }

  private renderTable(u: ViewUpdate): void {
    const ok = this.checkOrder(u);
    this.badge.textContent = ok ? '✓ sorted' : '✗ order';
    this.badge.className = `badge ${ok ? 'ok' : 'bad'}`;

    const cols = this.def.columns.map((n) => this.cols.byName.get(n)!);
    const table = el('table');
    const tr = el('tr');
    for (const c of cols) {
      const i = (this.spec.sort ?? []).findIndex((k) => k.column === c.id);
      const th = el('th', c.kind === ColumnKind.CATEGORY ? '' : 'num');
      th.textContent = c.name + (i >= 0 ? ` ${this.spec.sort![i].desc ? '↓' : '↑'}${i ? i + 1 : ''}` : '');
      if (c.derived) th.classList.add('derived');
      th.title = 'click: sort · shift+click: add secondary sort';
      th.onclick = (e) => this.setSort(c.id ?? 0, e.shiftKey);
      tr.append(th);
    }
    const thead = el('thead');
    thead.append(tr);
    const tbody = el('tbody');
    const next = new Map<number, number[]>();
    for (const r of u.rows ?? []) {
      const row = el('tr');
      const prev = this.last.get(r.row);
      for (const c of cols) {
        const v = r.values[c.id ?? 0] ?? 0;
        const td = el('td', cellClass(c, v), fmtCell(c, v));
        if (prev && prev[c.id ?? 0] !== v && c.kind !== ColumnKind.CATEGORY)
          td.classList.add(v > prev[c.id ?? 0] ? 'up' : 'down');
        row.append(td);
      }
      next.set(r.row, r.values);
      tbody.append(row);
    }
    this.last = next;
    table.append(thead, tbody);
    this.body.replaceChildren(table);
  }

  private renderPivot(cells: PivotCell[]): void {
    this.badge.textContent = '';
    const [n0, n1] = this.def.pivot!;
    const d0 = this.cols.byName.get(n0)!.dict, d1 = this.cols.byName.get(n1)!.dict;
    const m = d0.map(() => d1.map(() => 0));
    for (const c of cells) {
      const [a, b] = c.key ?? [];
      m[a ?? 0][b ?? 0] = Number(c[this.measure] ?? 0);
    }
    const rowTot = m.map((r) => r.reduce((s, v) => s + v, 0));
    const colTot = d1.map((_, j) => m.reduce((s, r) => s + r[j], 0));
    const max = Math.max(1, ...m.flat().map(Math.abs));

    const table = el('table', 'pivot');
    const hr = el('tr');
    hr.append(el('th', '', `${n0} \\ ${n1}`));
    for (const name of d1) hr.append(el('th', 'num', name));
    hr.append(el('th', 'num', 'total'));
    const thead = el('thead');
    thead.append(hr);
    const tbody = el('tbody');
    d0.forEach((name, i) => {
      const tr = el('tr');
      tr.append(el('th', '', name));
      m[i].forEach((v) => {
        const td = el('td', `num ${v < 0 ? 'neg' : ''}`, int.format(v));
        td.style.setProperty('--heat', String(Math.abs(v) / max));
        tr.append(td);
      });
      tr.append(el('td', 'num total', int.format(rowTot[i])));
      tbody.append(tr);
    });
    const ft = el('tr');
    ft.append(el('th', '', 'total'));
    for (const v of colTot) ft.append(el('td', 'num total', int.format(v)));
    ft.append(el('td', 'num total', int.format(rowTot.reduce((s, v) => s + v, 0))));
    tbody.append(ft);
    table.append(thead, tbody);
    this.body.replaceChildren(table);
  }
}

// ---------------------------------------------------------------- stats bar

class StatsBar {
  readonly root = el('div', 'stats');
  private readonly text = el('div', 'metrics');
  private readonly canvas = el('canvas');
  private readonly cpuBtn = el('button', '', 'CPU');
  private readonly gpuBtn = el('button', '', 'GPU');
  private readonly devices = el('div', 'devices');
  private readonly history: { ms: number; backend: Backend }[] = [];
  private schema: Schema | null = null;

  constructor(private readonly send: (m: ClientMessage) => void) {
    const toggle = el('div', 'toggle');
    toggle.append(el('span', 'label', 'engine'), this.cpuBtn, this.gpuBtn);
    this.cpuBtn.onclick = () => this.send({ set_backend: { backend: Backend.CPU } });
    this.gpuBtn.onclick = () => this.send({ set_backend: { backend: Backend.GPU } });
    this.canvas.width = 360;
    this.canvas.height = 56;
    this.root.append(toggle, this.text, this.canvas, this.devices);
  }

  setSchema(s: Schema): void {
    this.schema = s;
    this.gpuBtn.disabled = !(s.backends ?? []).includes(Backend.GPU);
    const gpus = s.gpu_names?.length ? s.gpu_names.join(' + ') : 'no GPU';
    this.devices.textContent = `${int.format(s.rows ?? 0)} orders · CPU ${s.cpu_threads ?? 0} threads · ${gpus}`;
  }

  update(st: Stats): void {
    const backend = st.backend ?? Backend.CPU;
    this.cpuBtn.classList.toggle('active', backend === Backend.CPU);
    this.gpuBtn.classList.toggle('active', backend === Backend.GPU);
    const ms = st.engine_ms ?? 0, budget = st.tick_ms ?? this.schema?.tick_ms ?? 300;
    this.history.push({ ms, backend });
    if (this.history.length > 180) this.history.shift();
    const val = st.validation ?? -1;
    const check = val < 0 ? '' : val === 0 ? '<span class="ok">CPU = GPU ✓</span>' : `<span class="bad">${val} views differ</span>`;
    const f1 = (x?: number) => (x ?? 0).toFixed(1);
    this.text.innerHTML =
      `<b class="${ms > budget ? 'bad' : ''}">${f1(ms)} ms</b> / ${budget} ms tick ` +
      `<small>(upload ${f1(st.upload_ms)} · derive ${f1(st.derive_ms)} · views ${f1(st.views_ms)})</small><br>` +
      `${st.views ?? 0} views (${st.client_views ?? 0} on screens + ${(st.views ?? 0) - (st.client_views ?? 0)} load) · ` +
      `${int.format(st.updates ?? 0)} cells changed · ${int.format(st.md_changes ?? 0)} prices moved ${check}`;
    this.draw(budget);
  }

  private draw(budget: number): void {
    const g = this.canvas.getContext('2d')!;
    const { width: w, height: h } = this.canvas;
    g.clearRect(0, 0, w, h);
    const top = Math.max(budget, ...this.history.map((p) => p.ms)) * 1.1;
    const y = (v: number) => h - (v / top) * h;
    g.strokeStyle = '#666';
    g.setLineDash([4, 4]);
    g.beginPath();
    g.moveTo(0, y(budget));
    g.lineTo(w, y(budget));
    g.stroke();
    g.setLineDash([]);
    const bw = w / 180;
    this.history.forEach((p, i) => {
      g.fillStyle = p.backend === Backend.GPU ? '#3ecf8e' : '#f5a623';
      g.fillRect(i * bw, y(p.ms), Math.max(1, bw - 0.5), h - y(p.ms));
    });
  }
}

// ---------------------------------------------------------------- app

class App {
  private ws: WebSocket | null = null;
  private panels: Panel[] = [];
  private readonly grid = el('main', 'grid');
  private readonly stats = new StatsBar((m) => this.send(m));
  private readonly url = el('input');
  private readonly conn = el('span', 'conn', 'disconnected');
  private readonly error = el('div', 'error');
  private wantOpen = false;

  constructor(root: HTMLElement) {
    const top = el('div', 'top');
    top.append(el('h1', '', 'GPU View Server'));
    let saved: string | null = null;
    try { saved = localStorage.getItem('gvs.url'); } catch { /* storage unavailable */ }
    const sameOrigin = location.port !== '5173' && location.protocol.startsWith('http');
    this.url.value = saved ?? (sameOrigin ? `ws://${location.host}/ws` : `ws://${location.hostname || 'localhost'}:8080/ws`);
    this.url.size = 32;
    const btn = el('button', '', 'Connect');
    btn.onclick = () => this.connect();
    top.append(this.url, btn, this.conn);
    root.append(top, this.stats.root, this.error, this.grid);
    this.connect();
  }

  send(m: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(encodeClient(m));
  }

  private connect(): void {
    this.wantOpen = true;
    try { localStorage.setItem('gvs.url', this.url.value); } catch { /* storage unavailable */ }
    this.ws?.close();
    const ws = new WebSocket(this.url.value);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    this.conn.textContent = 'connecting…';
    this.conn.className = 'conn';
    ws.onopen = () => {
      this.conn.textContent = 'connected';
      this.conn.className = 'conn ok';
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.conn.textContent = 'disconnected';
      this.conn.className = 'conn bad';
      if (this.wantOpen) setTimeout(() => this.ws === ws && this.connect(), 2000);
    };
    ws.onmessage = (e) => this.onMessage(new Uint8Array(e.data as ArrayBuffer));
  }

  private onMessage(buf: Uint8Array): void {
    const m = decodeServer(buf);
    switch (m.msg) {
      case 'schema': {
        const cols = new Cols(m.schema);
        this.stats.setSchema(m.schema);
        this.panels = PANELS.map((d, i) => new Panel(i + 1, d, cols, (x) => this.send(x)));
        this.grid.replaceChildren(...this.panels.map((p) => p.root));
        for (const p of this.panels) p.subscribe();
        break;
      }
      case 'view':
        this.panels.find((p) => p.id === m.view.view_id)?.render(m.view);
        break;
      case 'stats':
        this.stats.update(m.stats);
        break;
      case 'error':
        this.error.textContent = m.error.text;
        setTimeout(() => (this.error.textContent = ''), 5000);
        break;
    }
  }
}

new App(document.getElementById('app')!);
