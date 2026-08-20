# Combine devices with `MultiSource`

`MultiSource` is useful when one logical signal is physically distributed
across several devices, registers, or columns. A virtual map describes both how
to reach each source column and which metadata should be attached to it.

This example combines four digital inputs from two synthetic boards.

```python
import pandas as pd

from data_conduit.datasources.multisource import MultiSource

time = pd.Index([0.0, 0.5, 1.0, 1.5], name="Time")

board_a = pd.DataFrame(
    {
        "poke-0": [0, 1, 0, 0],
        "poke-1": [0, 0, 1, 0],
    },
    index=time,
)

board_b = pd.DataFrame(
    {
        "poke-2": [0, 0, 0, 1],
        "poke-3": [1, 0, 0, 0],
    },
    index=time,
)

source = MultiSource(
    dfs_dict={
        "board-a": {"digital": board_a},
        "board-b": {"digital": board_b},
    },
    virtual_maps={
        "activations": {
            "poke-0": {"device": "board-a", "register": "digital"},
            "poke-1": {"device": "board-a", "register": "digital"},
            "poke-2": {"device": "board-b", "register": "digital"},
            "poke-3": {"device": "board-b", "register": "digital"},
        }
    },
    global_coord_name="port",
    virtual_coord_names=["device", "register"],
    fill_value=0,
)

activations = source.data_arrays["activations"]
print(activations)
```

The resulting array has `Time` and `port` dimensions. `device` and `register`
are coordinates along `port`: they describe each logical port without adding
extra dense dimensions.

## Query virtual coordinates

The `.ulookup` accessor returns logical coordinate labels matching one or more
virtual-coordinate selectors:

```python
ports_on_a = activations.ulookup(device="board-a")
print(ports_on_a)
# ['poke-0', 'poke-1']
```

Use `.ulookup.select(...)` (or its `.sel(...)` alias) to return the corresponding
slice of the original array:

```python
board_a_only = activations.ulookup.select(device="board-a")
one_port = activations.ulookup.select(
    device="board-b",
    register="digital",
).sel(port="poke-2")
```

Selectors may be single values, lists, or inclusive slices. `None` is a
wildcard.

```python
selected = activations.ulookup.select(
    device=["board-a", "board-b"],
    register="digital",
)
```

## Keep the map declarative

The order of values inside each virtual-map entry is also the traversal order
through `dfs_dict`. In this example:

```text
dfs_dict[device][register][logical port column]
```

Use the same key order in every entry and pass `virtual_coord_names` explicitly
when clarity matters. HARP's `MultiDevice` and `Nosepoke` presets use this same
mechanism with device/register/channel mappings supplied for common rigs.
