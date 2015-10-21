#################
API Documentation
#################
This is a construction site...

*******
Network
*******

.. autoclass:: brainstorm.structure.network.Network
    :members:
    :special-members: __init__

*******
Trainer
*******
.. autoclass:: brainstorm.training.trainer.Trainer
    :members:
    :special-members: __init__

*****
Tools
*****
.. automodule:: brainstorm.tools
    :members:

**************
Data Iterators
**************
.. automodule:: brainstorm.data_iterators
    :members:

************
Initializers
************
.. automodule:: brainstorm.initializers
    :members:

*****
Hooks
*****
.. automodule:: brainstorm.hooks
    :members:

***************
Value Modifiers
***************

.. automodule:: brainstorm.value_modifiers
    :members:

*******
Scorers
*******
.. automodule:: brainstorm.scorers
    :members:

*******
Handler
*******
.. autoclass:: brainstorm.handlers.base_handler.Handler
    :members:
    :inherited-members:

************
Describables
************
.. automodule:: brainstorm.describable

Conversion to and from descriptions
===================================
.. autofunction:: create_from_description
.. autofunction:: get_description

Describable Base Class
======================
.. autoclass:: Describable
    :members:

    .. autoattribute:: __undescribed__
    .. autoattribute:: __default_values__
    .. automethod:: __init_from_description__
    .. automethod:: __describe__
    .. automethod:: __new_from_description__



