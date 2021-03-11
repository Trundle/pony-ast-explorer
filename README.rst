=================
Pony AST explorer
=================

A GDB plugin to interactively explore Pony_ ASTs.


Requirements
============

* GDB with Python support (Python >= 3.8)


How to use
==========

First, load the explorer inside a GDB session::

  source /path/to/checkout/ast_explorer.py

If the script was loaded successfully, there now exists a new interactive
command ``pony-ast``. Executing it will look for AST nodes in the current scope
and, if there are any, ask you which one to explore.

If you use AST explorer regularly, you might want to consider adding the source
line to your ``.gdbinit``.


Features
========

* The AST is drawn in more or less beatiful unicode boxes
* Show some details about a selected node
* Zoom out: set the parent node to the current root's parent
* Zoom in: set the currently selected node as new root node
* Fewer details: limit how far to descend into child nodes
* Jump to data: set a node's ``data`` field as new root node
* Print the currently selected node with either ``ast_print`` or
  ``ast_printverbose``.

Note that the mnemonics for the default keymap are currently a bit odd for some
commands. This is caused by my unusual keyboard layout and should probably be
changed.


Screenshot
==========

.. image:: https://trundle.github.io/pony-ast-explorer/ast-explorer.png


License
=======

AST explorer is released under the Apache License, Version 2.0. See ``LICENSE``
or http://www.apache.org/licenses/LICENSE-2.0.html for details.


.. _Pony: https://www.ponylang.io/
