import protobuf from 'protobufjs';
import protoText from '../../proto/gvs.proto?raw';

const root = protobuf.parse(protoText, { keepCase: true }).root;
const ClientMsg = root.lookupType('gvs.ClientMsg');
const ServerMsg = root.lookupType('gvs.ServerMsg');

export enum Op { EQ = 0, NE = 1, LT = 2, LE = 3, GT = 4, GE = 5, IN = 6 }
export enum Backend { CPU = 0, GPU = 1 }
export enum ColumnKind { CATEGORY = 0, UINT = 1, FLOAT = 2 }

export interface Predicate { column: number; op: Op; value?: number; mask?: number }
export interface SortKey { column: number; desc?: boolean }
export interface ViewSpec { filter?: Predicate[]; sort?: SortKey[]; offset?: number; limit?: number; pivot_by?: number[] }

export interface Column { id: number; name: string; kind: ColumnKind; dict: string[]; derived?: boolean }
export interface Schema {
  columns: Column[]; rows: number; backends: Backend[]; backend: Backend; tick_ms: number;
  gpu_names: string[]; cpu_threads: number; load_views: number;
}
export interface Row { row: number; values: number[] }
export interface PivotCell { key: number[]; count?: number; sum_leaves?: number; sum_notional?: number; sum_pnl?: number }
export interface ViewUpdate { view_id: number; tick: number; total?: number; offset?: number; rows: Row[]; pivot: PivotCell[] }
export interface Stats {
  tick: number; backend?: Backend; engine_ms?: number; derive_ms?: number; views_ms?: number; upload_ms?: number;
  views?: number; client_views?: number; rows?: number; updates?: number; md_changes?: number;
  validation?: number; tick_ms?: number;
}
export type ServerMessage =
  | { msg: 'schema'; schema: Schema }
  | { msg: 'view'; view: ViewUpdate }
  | { msg: 'stats'; stats: Stats }
  | { msg: 'error'; error: { text: string } };

export type ClientMessage =
  | { subscribe: { view_id: number; spec: ViewSpec } }
  | { unsubscribe: { view_id: number } }
  | { set_backend: { backend: Backend } };

export function encodeClient(msg: ClientMessage): Uint8Array {
  return ClientMsg.encode(ClientMsg.fromObject(msg)).finish();
}

export function decodeServer(buf: Uint8Array): ServerMessage {
  const obj = ServerMsg.toObject(ServerMsg.decode(buf), { longs: Number, enums: Number, arrays: true, oneofs: true });
  return obj as ServerMessage;
}
