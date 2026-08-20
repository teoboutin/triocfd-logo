"""Reading the LATA dumps a case wrote, for the elementary-tests form.

Everything the report measures is read through the master file, so no shape
and no file layout is hardcoded: the fields, the interface meshes, the domain
box and the element types are all declared there.

The `Format` line is honoured rather than assumed. Two files this form touches
disagree on it: the solver output is `LITTLE_ENDIAN,C_INDEXING`, while the
interface the decks start from, `src/init.lata`, is `ASCII,F_INDEXING`.
Reading the second with the conventions of the first gives triangles indexed
one vertex off, and a volume that is wrong without saying so.

The coordinate axes of an IJK domain come back through the same `REAL32`
output as everything else, so `cell_size` here is for saying whether the mesh
is uniform, never for computing a volume with - that number is read from the
deck by `decks.py`.
"""

import hashlib
import os
import re

import numpy as np

# The `key=value` attributes a `Geom` or a `Champ` line carries.
_KEYVAL = re.compile(r"(\w+)=([^\s]+)")


class Master:
    """The dumps a case wrote, read from its LATA master file.

    A master file is a `Format` line, a list of `Geom` lines naming an element
    type, and a list of `TEMPS` blocks each holding `Champ` lines that name a
    file, a geometry, a size and a component count. Both the Eulerian fields
    and the interface meshes are declared there.

    A `Champ` may override the file-level indexing with `NO_INDEXING`, which is
    what a field of per-element values carries.
    """

    def __init__(self, case_dir, case):
        self.dir = case_dir
        self.case = case
        self.path = os.path.join(case_dir, case + ".lata")
        self.byteorder = "<"
        self.ascii = False
        self.index_base = 0
        self.times = []
        self.elem_type = {}       # geometry name -> type_elem= of its Geom line
        self.blocks = []          # one dict {(name, geometry): entry} per TEMPS
        header = {}
        current = None
        with open(self.path) as handle:
            for line in handle:
                words = line.split()
                if not words:
                    continue
                if words[0] == "Format":
                    flags = words[1].upper()
                    self.ascii = "ASCII" in flags
                    if "BIG_ENDIAN" in flags:
                        self.byteorder = ">"
                    if "F_INDEXING" in flags:
                        self.index_base = 1
                elif words[0] == "Geom":
                    attrs = dict(_KEYVAL.findall(line))
                    self.elem_type[words[1]] = attrs.get("type_elem", "")
                elif words[0] == "TEMPS":
                    self.times.append(float(words[1]))
                    current = {}
                    self.blocks.append(current)
                elif words[0] == "Champ":
                    attrs = dict(_KEYVAL.findall(line))
                    entry = {
                        "file": words[2],
                        "size": int(attrs.get("size", 0)),
                        "components": int(attrs.get("composantes", 1)),
                        "format": attrs.get("format", attrs.get("FORMAT", "")),
                        "geometry": attrs.get("geometrie", ""),
                    }
                    key = (words[1], entry["geometry"])
                    (header if current is None else current)[key] = entry
        self.header = header

    # ------------------------------------------------------------- the arrays

    def path_of(self, entry):
        return os.path.join(self.dir, entry["file"])

    def raw_bytes(self, entry):
        """The bytes of one declared array, as they were written.

        Comparing these is the only way to claim that two runs wrote the same
        array: equality of the decoded values would still be an equality of
        numbers, and for a `REAL32` dump that is a weaker statement.
        """
        with open(self.path_of(entry), "rb") as handle:
            return handle.read()

    def digest(self, name, index, geometry="DOM"):
        """A digest of one stored array, or None where it was not dumped."""
        if not self.has(name, index, geometry):
            return None
        return hashlib.sha256(
            self.raw_bytes(self.entry(name, index, geometry))).hexdigest()

    def read(self, entry):
        """One declared array, as (size, components), in double precision."""
        if self.ascii:
            raw = np.loadtxt(self.path_of(entry))
        else:
            dtype = "i4" if "INT32" in entry["format"].upper() else "f4"
            raw = np.fromfile(self.path_of(entry), dtype=self.byteorder + dtype)
        return raw.reshape(entry["size"], entry["components"])

    # ------------------------------------------------------------ the domain

    def node_coordinates(self):
        """The three coordinate axes of the IJK domain, in m.

        An IJK structured domain declares its geometry as three axes rather
        than as a node list plus a connectivity, so the domain box and the cell
        sizes are read from these and not inferred.
        """
        return [self.read(self.header[(f"SOMMETS_IJK_{axis}", "DOM")]).ravel()
                .astype(float) for axis in "IJK"]

    def cell_size(self):
        """(dx, dy, dz) in m, and whether each axis is uniform."""
        sizes, uniform = [], True
        for axis in self.node_coordinates():
            steps = np.diff(axis)
            sizes.append(float(steps.mean()))
            uniform = uniform and bool(np.allclose(steps, steps[0], rtol=1e-6))
        return sizes, uniform

    def domain_origin(self):
        """The low corner of the domain box, in m."""
        return [float(axis[0]) for axis in self.node_coordinates()]

    def domain_size(self):
        """The domain extent along each axis, in m."""
        return [float(axis[-1] - axis[0]) for axis in self.node_coordinates()]

    # ------------------------------------------------------------- the dumps

    def has(self, name, index, geometry="DOM"):
        return (0 <= index < len(self.blocks)
                and (name, geometry) in self.blocks[index])

    def entry(self, name, index, geometry="DOM"):
        return self.blocks[index][(name, geometry)]

    def field(self, name, index, geometry="DOM"):
        """One declared field, flattened, in double precision."""
        return self.read(self.entry(name, index, geometry)).ravel().astype(float)

    def mesh(self, index, geometry="INTERFACES"):
        """(nodes, triangles) of an interface geometry at one dump.

        The nodes are absolute coordinates in m; the triangles come back
        zero-based whatever the file declares.
        """
        block = self.blocks[index]
        nodes = self.read(block[("SOMMETS", geometry)]).astype(float)
        entry = block[("ELEMENTS", geometry)]
        triangles = self.read(entry).astype(np.int64)
        if "NO_INDEXING" not in entry["format"].upper():
            triangles = triangles - self.index_base
        return nodes, triangles

    def complete(self, index):
        """Whether one dump carries both an `INDICATRICE` field and a mesh.

        Every measurement of this form needs both at the same instant, so a
        dump missing either is skipped rather than raised on.
        """
        return (self.has("INDICATRICE", index)
                and self.has("SOMMETS", index, "INTERFACES"))


def open_case(build_dir, case):
    """The master file of one case, or None where the case wrote none.

    A case that ran but wrote no LATA is a result the report states, not an
    exception: a form that raises produces no report at all.
    """
    try:
        return Master(os.path.join(build_dir, case), case)
    except OSError:
        return None
