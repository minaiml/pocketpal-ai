import type {
  ListDerivedCaps,
  RemoteModelInfo,
  ServerConfig,
} from '../../utils/types';
import {dialectFor} from './index';

/**
 * Read a models-list row for capabilities, through the dialect of the
 * persisted server type. Pure, and never a default: anything absent, wrongly
 * typed or unparseable yields no field at all, so a caller can tell "this
 * server says no" from "this server did not say".
 */
export function deriveListCaps(
  row: RemoteModelInfo | undefined,
  serverType: string | undefined,
): ListDerivedCaps {
  return dialectFor(serverType).readModelEntry(row);
}

/**
 * The same derivation for every model of every server, keyed as
 * `${serverId}/${remoteModelId}` — the id a remote `Model` carries.
 */
export function deriveListCapsMap(
  servers: ServerConfig[],
  serverModels: {get(serverId: string): RemoteModelInfo[] | undefined},
): Record<string, ListDerivedCaps> {
  const map: Record<string, ListDerivedCaps> = {};
  for (const server of servers) {
    for (const row of serverModels.get(server.id) ?? []) {
      map[`${server.id}/${row.id}`] = deriveListCaps(row, server.serverType);
    }
  }
  return map;
}
