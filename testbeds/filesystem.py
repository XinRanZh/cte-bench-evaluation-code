"""
Testbed 6: In-memory filesystem with inodes, locks, and timeouts.

Algorithm ported from pyfilesystem2 MemoryFS (PyFilesystem/pyfilesystem2,
master branch) — fs/memoryfs.py: makedir / openbin / remove / removedir /
getinfo / listdir semantics. Upstream uses threading.RLock for per-entry
concurrency; we keep the lock as logical state so a counterfactual on the
lock table is well-defined. Directory / file / empty-directory error names
mirror upstream fs.errors.

Upstream reference: https://github.com/PyFilesystem/pyfilesystem2
License: MIT (upstream). This port: research benchmark only.

Invariants:
  INV-1: a locked inode can only be released by its holder, and exactly once
  INV-2: removedir on non-empty directory is rejected (DirectoryNotEmpty)
  INV-3: inode_id is unique across all history
  INV-4: a file lock expires on advance_clock past its hold_until tick
  INV-5: open_bin with mode "w" on an existing directory is rejected
"""
from __future__ import annotations

SOURCE_CODE = '''
class FileSystemService:
    """In-memory filesystem with inodes, locks, and lock-TTL expiry.
    Ported from pyfilesystem2.memoryfs.MemoryFS.
    """

    def __init__(self):
        # inode_id "i<N>" -> {"kind":"file"|"dir", "parent":path, "name":str,
        #                     "children":[inode_ids] for dir, "size":int for file}
        self.inodes: dict[str, dict] = {}
        self.root_id: str = "i0"
        self.inodes[self.root_id] = {"kind": "dir", "parent": None,
                                     "name": "/", "children": []}
        self.locks: dict[str, dict] = {}  # inode_id -> {holder, hold_until}
        self.clock = 0
        self.counter = 0

    def _path_exists(self, parent_id, name):
        if parent_id not in self.inodes:
            return False
        return any(self.inodes[c]["name"] == name
                   for c in self.inodes[parent_id]["children"])

    def _child(self, parent_id, name):
        for cid in self.inodes[parent_id]["children"]:
            if self.inodes[cid]["name"] == name:
                return cid
        return None

    def advance_clock(self, delta):
        if delta < 0:
            return {"error": {"code": "NEGATIVE_DELTA"}}
        self.clock += delta
        released = []
        for iid, l in list(self.locks.items()):
            if l["hold_until"] <= self.clock:
                self.locks.pop(iid)
                released.append(iid)
        return {"clock": self.clock, "released": released}

    def makedir(self, parent_id, name):
        if parent_id not in self.inodes or self.inodes[parent_id]["kind"] != "dir":
            return {"error": {"code": "PARENT_NOT_DIR", "message": parent_id}}
        if self._path_exists(parent_id, name):
            return {"error": {"code": "DIRECTORY_EXISTS", "message": name}}
        self.counter += 1
        iid = f"i{self.counter}"
        self.inodes[iid] = {"kind": "dir", "parent": parent_id,
                            "name": name, "children": []}
        self.inodes[parent_id]["children"].append(iid)
        return {"inode_id": iid, "name": name}

    def openbin(self, parent_id, name, mode):
        if parent_id not in self.inodes or self.inodes[parent_id]["kind"] != "dir":
            return {"error": {"code": "PARENT_NOT_DIR", "message": parent_id}}
        existing = self._child(parent_id, name)
        if mode == "w":
            if existing is not None and self.inodes[existing]["kind"] == "dir":
                return {"error": {"code": "FILE_EXPECTED", "message": name}}
            if existing is None:
                self.counter += 1
                iid = f"i{self.counter}"
                self.inodes[iid] = {"kind": "file", "parent": parent_id,
                                    "name": name, "size": 0}
                self.inodes[parent_id]["children"].append(iid)
                return {"inode_id": iid, "name": name, "mode": "w", "size": 0}
            self.inodes[existing]["size"] = 0
            return {"inode_id": existing, "name": name, "mode": "w", "size": 0}
        if mode == "r":
            if existing is None:
                return {"error": {"code": "RESOURCE_NOT_FOUND", "message": name}}
            if self.inodes[existing]["kind"] != "file":
                return {"error": {"code": "FILE_EXPECTED", "message": name}}
            return {"inode_id": existing, "name": name, "mode": "r",
                    "size": self.inodes[existing]["size"]}
        return {"error": {"code": "UNKNOWN_MODE", "message": mode}}

    def write_bytes(self, inode_id, n):
        if inode_id not in self.inodes:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": inode_id}}
        if self.inodes[inode_id]["kind"] != "file":
            return {"error": {"code": "FILE_EXPECTED", "message": inode_id}}
        if n < 0:
            return {"error": {"code": "NEGATIVE_BYTES", "message": str(n)}}
        if inode_id in self.locks:
            return {"error": {"code": "LOCKED", "message": self.locks[inode_id]["holder"]}}
        self.inodes[inode_id]["size"] += n
        return {"inode_id": inode_id, "size": self.inodes[inode_id]["size"]}

    def remove(self, inode_id):
        if inode_id == self.root_id:
            return {"error": {"code": "CANNOT_REMOVE_ROOT"}}
        if inode_id not in self.inodes:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": inode_id}}
        node = self.inodes[inode_id]
        if node["kind"] == "dir":
            if node["children"]:
                return {"error": {"code": "DIRECTORY_NOT_EMPTY", "message": inode_id}}
        if inode_id in self.locks:
            return {"error": {"code": "LOCKED", "message": self.locks[inode_id]["holder"]}}
        parent = self.inodes[node["parent"]]
        parent["children"].remove(inode_id)
        self.inodes.pop(inode_id)
        return {"removed": inode_id}

    def lock(self, inode_id, holder, ttl):
        if inode_id not in self.inodes:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": inode_id}}
        if ttl <= 0:
            return {"error": {"code": "INVALID_TTL", "message": str(ttl)}}
        if inode_id in self.locks:
            return {"error": {"code": "ALREADY_LOCKED",
                              "message": self.locks[inode_id]["holder"]}}
        self.locks[inode_id] = {"holder": holder, "hold_until": self.clock + ttl}
        return {"inode_id": inode_id, "holder": holder,
                "hold_until": self.clock + ttl}

    def unlock(self, inode_id, holder):
        if inode_id not in self.locks:
            return {"error": {"code": "NOT_LOCKED", "message": inode_id}}
        if self.locks[inode_id]["holder"] != holder:
            return {"error": {"code": "WRONG_HOLDER",
                              "message": self.locks[inode_id]["holder"]}}
        self.locks.pop(inode_id)
        return {"inode_id": inode_id, "unlocked": True}

    def listdir(self, inode_id):
        if inode_id not in self.inodes:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": inode_id}}
        if self.inodes[inode_id]["kind"] != "dir":
            return {"error": {"code": "DIRECTORY_EXPECTED", "message": inode_id}}
        names = sorted(self.inodes[c]["name"]
                       for c in self.inodes[inode_id]["children"])
        return {"inode_id": inode_id, "names": names}
'''


class FileSystemService:
    def __init__(self):
        self.inodes: dict[str, dict] = {}
        self.root_id = "i0"
        self.inodes[self.root_id] = {"kind": "dir", "parent": None,
                                     "name": "/", "children": []}
        self.locks: dict[str, dict] = {}
        self.clock = 0
        self.counter = 0

    def _path_exists(self, parent_id, name):
        if parent_id not in self.inodes:
            return False
        return any(self.inodes[c]["name"] == name
                   for c in self.inodes[parent_id]["children"])

    def _child(self, parent_id, name):
        for cid in self.inodes[parent_id]["children"]:
            if self.inodes[cid]["name"] == name:
                return cid
        return None

    def advance_clock(self, delta):
        if delta < 0:
            return {"error": {"code": "NEGATIVE_DELTA"}}
        self.clock += delta
        released = []
        for iid, l in list(self.locks.items()):
            if l["hold_until"] <= self.clock:
                self.locks.pop(iid)
                released.append(iid)
        return {"clock": self.clock, "released": released}

    def makedir(self, parent_id, name):
        if parent_id not in self.inodes or self.inodes[parent_id]["kind"] != "dir":
            return {"error": {"code": "PARENT_NOT_DIR", "message": parent_id}}
        if self._path_exists(parent_id, name):
            return {"error": {"code": "DIRECTORY_EXISTS", "message": name}}
        self.counter += 1
        iid = f"i{self.counter}"
        self.inodes[iid] = {"kind": "dir", "parent": parent_id,
                            "name": name, "children": []}
        self.inodes[parent_id]["children"].append(iid)
        return {"inode_id": iid, "name": name}

    def openbin(self, parent_id, name, mode):
        if parent_id not in self.inodes or self.inodes[parent_id]["kind"] != "dir":
            return {"error": {"code": "PARENT_NOT_DIR", "message": parent_id}}
        existing = self._child(parent_id, name)
        if mode == "w":
            if existing is not None and self.inodes[existing]["kind"] == "dir":
                return {"error": {"code": "FILE_EXPECTED", "message": name}}
            if existing is None:
                self.counter += 1
                iid = f"i{self.counter}"
                self.inodes[iid] = {"kind": "file", "parent": parent_id,
                                    "name": name, "size": 0}
                self.inodes[parent_id]["children"].append(iid)
                return {"inode_id": iid, "name": name, "mode": "w", "size": 0}
            self.inodes[existing]["size"] = 0
            return {"inode_id": existing, "name": name, "mode": "w", "size": 0}
        if mode == "r":
            if existing is None:
                return {"error": {"code": "RESOURCE_NOT_FOUND", "message": name}}
            if self.inodes[existing]["kind"] != "file":
                return {"error": {"code": "FILE_EXPECTED", "message": name}}
            return {"inode_id": existing, "name": name, "mode": "r",
                    "size": self.inodes[existing]["size"]}
        return {"error": {"code": "UNKNOWN_MODE", "message": mode}}

    def write_bytes(self, inode_id, n):
        if inode_id not in self.inodes:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": inode_id}}
        if self.inodes[inode_id]["kind"] != "file":
            return {"error": {"code": "FILE_EXPECTED", "message": inode_id}}
        if n < 0:
            return {"error": {"code": "NEGATIVE_BYTES", "message": str(n)}}
        if inode_id in self.locks:
            return {"error": {"code": "LOCKED", "message": self.locks[inode_id]["holder"]}}
        self.inodes[inode_id]["size"] += n
        return {"inode_id": inode_id, "size": self.inodes[inode_id]["size"]}

    def remove(self, inode_id):
        if inode_id == self.root_id:
            return {"error": {"code": "CANNOT_REMOVE_ROOT"}}
        if inode_id not in self.inodes:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": inode_id}}
        node = self.inodes[inode_id]
        if node["kind"] == "dir":
            if node["children"]:
                return {"error": {"code": "DIRECTORY_NOT_EMPTY", "message": inode_id}}
        if inode_id in self.locks:
            return {"error": {"code": "LOCKED", "message": self.locks[inode_id]["holder"]}}
        parent = self.inodes[node["parent"]]
        parent["children"].remove(inode_id)
        self.inodes.pop(inode_id)
        return {"removed": inode_id}

    def lock(self, inode_id, holder, ttl):
        if inode_id not in self.inodes:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": inode_id}}
        if ttl <= 0:
            return {"error": {"code": "INVALID_TTL", "message": str(ttl)}}
        if inode_id in self.locks:
            return {"error": {"code": "ALREADY_LOCKED",
                              "message": self.locks[inode_id]["holder"]}}
        self.locks[inode_id] = {"holder": holder, "hold_until": self.clock + ttl}
        return {"inode_id": inode_id, "holder": holder,
                "hold_until": self.clock + ttl}

    def unlock(self, inode_id, holder):
        if inode_id not in self.locks:
            return {"error": {"code": "NOT_LOCKED", "message": inode_id}}
        if self.locks[inode_id]["holder"] != holder:
            return {"error": {"code": "WRONG_HOLDER",
                              "message": self.locks[inode_id]["holder"]}}
        self.locks.pop(inode_id)
        return {"inode_id": inode_id, "unlocked": True}

    def listdir(self, inode_id):
        if inode_id not in self.inodes:
            return {"error": {"code": "RESOURCE_NOT_FOUND", "message": inode_id}}
        if self.inodes[inode_id]["kind"] != "dir":
            return {"error": {"code": "DIRECTORY_EXPECTED", "message": inode_id}}
        names = sorted(self.inodes[c]["name"]
                       for c in self.inodes[inode_id]["children"])
        return {"inode_id": inode_id, "names": names}

    def state_dict(self):
        return {"inodes": {iid: {k: (list(v) if isinstance(v, list) else v)
                                 for k, v in n.items()}
                           for iid, n in self.inodes.items()},
                "locks": {iid: dict(l) for iid, l in self.locks.items()},
                "clock": self.clock, "counter": self.counter,
                "root_id": self.root_id}


OPS = [
    ("advance_clock", ["delta"]),
    ("makedir",       ["parent_id", "name"]),
    ("openbin",       ["parent_id", "name", "mode"]),
    ("write_bytes",   ["inode_id", "n"]),
    ("remove",        ["inode_id"]),
    ("lock",          ["inode_id", "holder", "ttl"]),
    ("unlock",        ["inode_id", "holder"]),
    ("listdir",       ["inode_id"]),
]


def check_invariants(state: dict) -> list[str]:
    viols = []
    inodes = state["inodes"]
    locks = state["locks"]
    # INV-3: unique inode ids. The dict keys enforce this, so a violation here
    # would require manual state injection.
    iids = list(inodes.keys())
    if len(iids) != len(set(iids)):
        viols.append("INV3_DUP_INODE")
    # INV-4: no lock is retained past hold_until <= clock.
    clock = state["clock"]
    for iid, l in locks.items():
        if l["hold_until"] <= clock:
            viols.append(f"INV4_EXPIRED_LOCK_LIVE:{iid}:until={l['hold_until']}<=clock={clock}")
    # INV-5 structural: every locked inode must exist in inodes.
    for iid in locks:
        if iid not in inodes:
            viols.append(f"INV_STRUCTURAL_LOCK_ORPHAN:{iid}")
    # parent integrity: every child is listed under its parent.
    for iid, n in inodes.items():
        if n["parent"] is None:
            continue
        if n["parent"] not in inodes:
            viols.append(f"INV_STRUCTURAL_ORPHAN:{iid}:parent={n['parent']}")
            continue
        if iid not in inodes[n["parent"]].get("children", []):
            viols.append(f"INV_STRUCTURAL_CHILD_MISSING:{iid}")
    return viols


NAME = "filesystem"


def new_service():
    return FileSystemService()


# State-probe v2: testbed-level predeclared paths.
PROBE_PATHS_V2 = [
    ["clock"],
    ["counter"],
    ["root_id"],
    ("dynamic", "inode_count",
     lambda s: [["__len__", "inodes"]]),
    ("dynamic", "lock_count",
     lambda s: [["__len__", "locks"]]),
    ("dynamic", "inode_kinds",
     lambda s: [["inodes", i, "kind"] for i in (s.get("inodes") or {})]),
]
