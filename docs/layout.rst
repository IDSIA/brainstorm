======
Layout
======
Layouts describe how the memory for the network should be arranged.
We use the following network as an example here:

.. code-block:: python

    mse = MseLayer(10)
    InputLayer(4) >> RnnLayer(5) >> FullyConnectedLayer(10, name='OutLayer') >> mse.net_out
    DataLayer(10, data_from='mse_targets') >> mse.targets
    net = build_net(mse)


Parameters
==========
The parameter layout is essentially a tree, describing how a chunk of memory
should be split up into parameters, and how to name and shape them:

.. code-block:: python

    param_layout = {
        'slice': (0, 110),
        'structure': [
            ('InputLayer', {
                'slice': (0, 0),
                'structure': []
            }),
            ('DataLayer', {
                'slice': (0, 0),
                'structure': []
            }),
            ('RnnLayer', {
                'slice': (0, 50),
                'structure': [
                    ('W', {'slice': (0, 20),  'shape': (4, 5)}),
                    ('R', {'slice': (20, 45), 'shape': (5, 5)}),
                    ('b', {'slice': (45, 50), 'shape': (5,  )})
                ]
            }),
            ('OutLayer', {
                'slice': (50, 110),
                'structure': [
                    ('W', {'slice': (50, 100),  'shape': (5, 10)}),
                    ('b', {'slice': (100, 110), 'shape': (10,  )})
                ]
            }),
            ('MseLayer', {
                'slice': (0, 0),
                'structure': []
            })
        ]
    }


In/Out
======


.. code-block:: python

    inputs_layout = {
        'slice': (0, 29),
        'structure': [
            ('InputLayer', {
                'slice': (0, 0),
                'structure': []
            }),
            ('DataLayer', {
                'slice': (0, 0),
                'structure': []
            }),
            ('RnnLayer', {
                'slice': (0, 4),
                'structure': [
                    ('default', {'slice': (0, 4),  'shape': (4,)})
                ]
            }),
            ('OutLayer', {
                'slice': (14, 19),
                'structure': [
                    ('default', {'slice': (14, 19),  'shape': (5,)})
                ]
            }),
            ('MseLayer', {
                # slice is missing, because memory region is disconnected
                'structure': [
                    ('net_out', {'slice': (19, 29),  'shape': (10,)}),
                    ('targets', {'slice': (4,  14),  'shape': (10,)})
                ]
            })
        ]
    }



.. code-block:: python

    outputs_layout = {
        'slice': (0, 30),
        'structure': [
            ('InputLayer', {
                'slice': (0, 4),
                'structure': [
                    ('default', {'slice': (0, 4),  'shape': (4,)})
                ]
            }),
            ('DataLayer', {
                'slice': (4, 14),
                'structure': [
                    ('default', {'slice': (4, 14),  'shape': (10,)})
                ]
            }),
            ('RnnLayer', {
                'slice': (14, 19),
                'structure': [
                    ('default', {'slice': (14, 19),  'shape': (5,)})
                ]
            }),
            ('OutLayer', {
                'slice': (19, 29),
                'structure': [
                    ('default', {'slice': (19, 29),  'shape': (10,)})
                ]
            }),
            ('MseLayer', {
                'slice': (29, 30),
                'structure': [
                    ('default', {'slice': (29, 30),  'shape': (1,)})
                ]
            })
        ]
    }





Internal State
==============

.. code-block:: python

    state_layout = {
        'slice': (0, 15),
        'structure': [
            ('InputLayer', {
                'slice': (0, 0),
                'structure': []
            }),
            ('DataLayer', {
                'slice': (0, 0),
                'structure': []
            }),
            ('RnnLayer', {
                'slice': (0, 5),
                'structure': [
                    ('Ha', {'slice': (0, 5),  'shape': (5,)})
                ]
            }),
            ('OutLayer', {
                'slice': (5, 15),
                'structure': [
                    ('Ha', {'slice': (5, 15),  'shape': (10,)})
                ]
            }),
            ('MseLayer', {
                'slice': (15, 15),
                'structure': []
            })
        ]
    }


Joint Layout
============

.. code-block:: python

    joint_layout = {
        't_slice': (0, 45),
        'c_slice': (0, 110),
        'structure': [
            ('InputLayer', {
                'structure': [
                    ('parameters', {'structure': []}),
                    ('inputs', {'structure': []}),
                    ('outputs', {'t_slice': (0, 4), 'structure': [
                        ('default', {'t_slice': (0, 4),  'shape': (4,)})
                    ]}),
                    ('state', {'structure': []}),
                ]
            }),
            ('DataLayer', {
                'structure': [
                    ('parameters', {'structure': []}),
                    ('inputs', {'structure': []}),
                    ('outputs', {'t_slice': (10, 14), 'structure': [
                        ('default', {'t_slice': (10, 14),  'shape': (4,)})
                    ]}),
                    ('state', {'structure': []}),
                ]
            }),
            ('RnnLayer', {
                'structure': [
                    ('parameters', {'c_slice': (0, 50), 'structure': [
                        ('W', {'slice': (0, 20),  'shape': (4, 5)}),
                        ('R', {'slice': (20, 45), 'shape': (5, 5)}),
                        ('b', {'slice': (45, 50), 'shape': (5,  )})
                    ]}),
                    ('inputs', {'t_slice': (0, 4), 'structure': [
                        ('default', {'t_slice': (0, 4),  'shape': (4,)})
                    ]}),
                    ('outputs', {'t_slice': (14, 19), 'structure': [
                        ('default', {'t_slice': (14, 19),  'shape': (5,)})
                    ]}),
                    ('state', {'t_slice': (30, 35), 'structure': [
                        ('Ha', {'t_slice': (30, 35),  'shape': (5,)})
                    ]}),
                ]
            }),
            ('OutLayer', {
                'structure': [
                    ('parameters', {'c_slice': (50, 110), 'structure': [
                        ('W', {'c_slice': (50, 100),  'shape': (5, 10)}),
                        ('b', {'c_slice': (100, 110), 'shape': (10,  )})
                    ]}),
                    ('inputs', {'t_slice': (14, 19), 'structure': [
                        ('default', {'t_slice': (14, 19),  'shape': (5,)})
                    ]}),
                    ('outputs', {'t_slice': (19, 29), 'structure': [
                        ('default', {'t_slice': (19, 29),  'shape': (10,)})
                    ]}),
                    ('state', {'t_slice': (35, 45), 'structure': [
                        ('Ha', {'t_slice': (35, 55),  'shape': (10,)})
                    ]}),
                ]
            }),
            ('MseLayer', {
                'slice': (0, 0),
                'structure': [
                    ('parameters', {'structure': []}),
                    ('inputs', {'structure': [
                        ('net_out', {'t_slice': (19, 29),  'shape': (10,)}),
                        ('targets', {'t_slice': (10, 14),  'shape': (10,)}),
                    ]}),
                    ('outputs', {'t_slice': (29, 30), 'structure': [
                        ('default', {'t_slice': (29, 30),  'shape': (1,)})
                    ]}),
                    ('state', {'structure': []}),
                ]
            })
        ]
    }