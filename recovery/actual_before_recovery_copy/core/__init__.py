"""Core algorithms grouped into lightweight, independently imported subpackages.

Public functions are exported by their owning subpackage. This namespace avoids
eager aggregation so importing a core helper does not load unrelated optional
integration or visualisation dependencies.
"""
