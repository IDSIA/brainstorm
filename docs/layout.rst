======
Layout
======
Layouts describe how the memory for the network should be arranged.
We use the following network as an example here:

.. code-block:: python

    mse = MseLayer(10)
    DataLayer(4) - 'input_data' >> RnnLayer(5) >> FullyConnectedLayer(10, name='OutLayer') >> 'net_out' - mse
    DataLayer(10) - 'targets' >> 'targets' - mse
    net = build_net(mse)



Joint Layout
============

.. code-block:: python

    joint_layout = {
        't_slice': (0, 45),
        'b_slice': (0, 0),
        'c_slice': (0, 110),
        'structure': [
            ('DataLayer', {
                'structure': [
                    ('parameters', {'structure': []}),
                    ('inputs', {'structure': []}),
                    ('outputs', {'t_slice': (0, 14), 'structure': [
                        ('input_data', {'t_slice': (0, 4),  'shape': (4,)})
                        ('targets', {'t_slice': (10, 14),  'shape': (4,)})
                    ]}),
                    ('state', {'structure': []}),
                ]
            }),
            ('RnnLayer', {
                'structure': [
                    ('parameters', {'c_slice': (0, 50), 'structure': [
                        ('W', {'c_slice': (0, 20),  'shape': (4, 5)}),
                        ('R', {'c_slice': (20, 45), 'shape': (5, 5)}),
                        ('b', {'c_slice': (45, 50), 'shape': (5,  )})
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