#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.network import Network
import numpy as np
from six import string_types

from collections import OrderedDict
from brainstorm.describable import Describable
from brainstorm.training.trainer import run_network
from brainstorm.utils import get_by_path


class Hook(Describable):
    __undescribed__ = {
        '__name__',  # the name is saved in the trainer
        'run_verbosity'
    }
    __default_values__ = {
        'timescale': 'epoch',
        'interval': 1,
        'verbose': None
    }

    def __init__(self, name=None, timescale='epoch', interval=1, verbose=None):
        self.timescale = timescale
        self.interval = interval
        self.__name__ = name or self.__class__.__name__
        self.priority = 0
        self.verbose = verbose
        self.run_verbosity = None

    def start(self, net, stepper, verbose, monitor_kwargs):
        if self.verbose is None:
            self.run_verbosity = verbose
        else:
            self.run_verbosity = self.verbose

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        pass


class SaveNetwork(Hook):
    """
    Save the weights of the network to the given file on every call.
    Default is to save them once per epoch, but this can be configured using
    the timescale and interval parameters.
    """

    def __init__(self, filename, name=None, timescale='epoch', interval=1):
        super(SaveNetwork, self).__init__(name, timescale, interval)
        self.filename = filename

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        net.save_as_hdf5(self.filename)

    def load_network(self):
        return Network.from_hdf5(self.filename)


class SaveBestNetwork(Hook):
    """
    Check every epoch to see if the given objective is at it's best value and
    if so, save the network to the specified file.
    """
    __undescribed__ = {'parameters': None}
    __default_values__ = {'filename': None}

    def __init__(self, log_name, filename=None, name=None,
                 criterion='max', verbose=None):
        super(SaveBestNetwork, self).__init__(name, 'epoch', 1, verbose)
        self.log_name = log_name
        self.filename = filename
        self.parameters = None
        assert criterion == 'min' or criterion == 'max'
        self.criterion = criterion

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        e = get_by_path(logs, self.log_name)
        best_idx = np.argmin(e) if self.criterion == 'min' else np.argmax(e)
        if best_idx == len(e) - 1:
            params = net.handler.get_numpy_copy(net.buffer.parameters)
            if self.filename is not None:
                if self.run_verbosity:
                    print("{} >> {} improved. Saving network to {} ...".format(
                          self.__name__, self.log_name, self.filename))
                net.save_as_hdf5(self.filename)
            else:
                if self.run_verbosity:
                    print("{} >> {} improved. Caching parameters ...".format(
                          self.__name__, self.log_name))
                self.parameters = params
        elif self.run_verbosity:
            print("{} >> Last saved parameters after epoch {} when {} was {}"
                  "".format(self.__name__, best_idx, self.log_name,
                            e[best_idx]))

    def load_parameters(self):
        return np.load(self.filename) if self.filename is not None \
            else self.parameters


class MonitorLayerParameters(Hook):
    """
    Monitor some properties of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        if name is None:
            name = "MonitorParameters_{}".format(layer_name)
        super(MonitorLayerParameters, self).__init__(name, timescale,
                                                     interval, verbose)
        self.layer_name = layer_name

    def start(self, net, stepper, verbose, monitor_kwargs):
        assert self.layer_name in net.layers.keys(), \
            "{} >> No layer named {} present in network. Available layers " \
            "are {}.".format(self.__name__, self.layer_name, net.layers.keys())

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        log = OrderedDict()
        for key, v in net.buffer[self.layer_name].parameters.items():
            v = net.handler.get_numpy_copy(v)
            log[key] = OrderedDict()
            log[key]['min'] = v.min()
            log[key]['avg'] = v.mean()
            log[key]['max'] = v.max()
            if len(v.shape) > 1:
                log[key]['min_L2_norm'] = np.sqrt(np.sum(v ** 2, axis=1)).min()
                log[key]['avg_L2_norm'] = np.sqrt(np.sum(v ** 2,
                                                         axis=1)).mean()
                log[key]['max_L2_norm'] = np.sqrt(np.sum(v ** 2, axis=1)).max()

        return log


class MonitorLayerGradients(Hook):
    """
    Monitor some statistics about all the gradients of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        if name is None:
            name = "MonitorGradients_{}".format(layer_name)
        super(MonitorLayerGradients, self).__init__(name, timescale,
                                                    interval, verbose)
        self.layer_name = layer_name

    def start(self, net, stepper, verbose, monitor_kwargs):
        assert self.layer_name in net.layers.keys(), \
            "{} >> No layer named {} present in network. Available layers " \
            "are {}.".format(self.__name__, self.layer_name, net.layers.keys())

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        log = OrderedDict()
        for key, v in net.buffer[self.layer_name].gradients.items():
            v = net.handler.get_numpy_copy(v)
            log[key] = OrderedDict()
            log[key]['min'] = v.min()
            log[key]['avg'] = v.mean()
            log[key]['max'] = v.max()
        return log


class MonitorLayerDeltas(Hook):
    """
    Monitor some statistics about all the deltas of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        if name is None:
            name = "MonitorDeltas_{}".format(layer_name)
        super(MonitorLayerDeltas, self).__init__(name, timescale,
                                                 interval, verbose)
        self.layer_name = layer_name

    def start(self, net, stepper, verbose, monitor_kwargs):
        assert self.layer_name in net.layers.keys(), \
            "{} >> No layer named {} present in network. Available layers " \
            "are {}.".format(self.__name__, self.layer_name, net.layers.keys())

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        log = OrderedDict()
        for key, v in net.buffer[self.layer_name].internals.items():
            v = net.handler.get_numpy_copy(v)
            log[key] = OrderedDict()
            log[key]['min'] = v.min()
            log[key]['avg'] = v.mean()
            log[key]['max'] = v.max()

        out_deltas_log = log['output_deltas'] = OrderedDict()
        for key, v in net.buffer[self.layer_name].output_deltas.items():
            v = net.handler.get_numpy_copy(v)
            key_log = out_deltas_log[key] = OrderedDict()
            key_log['min'] = v.min()
            key_log['avg'] = v.mean()
            key_log['max'] = v.max()

        in_deltas_log = log['input_deltas'] = OrderedDict()
        for key, v in net.buffer[self.layer_name].input_deltas.items():
            key_log = in_deltas_log[key] = OrderedDict()
            v = net.handler.get_numpy_copy(v)
            key_log[key]['min'] = v.min()
            key_log[key]['avg'] = v.mean()
            key_log[key]['max'] = v.max()

        return log


class MonitorLayerInOuts(Hook):
    """
    Monitor some statistics about all the inputs and outputs of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        if name is None:
            name = "MonitorInOuts_{}".format(layer_name)
        super(MonitorLayerInOuts, self).__init__(name, timescale,
                                                 interval, verbose)
        self.layer_name = layer_name

    def start(self, net, stepper, verbose, monitor_kwargs):
        assert self.layer_name in net.layers.keys(), \
            "{} >> No layer named {} present in network. Available layers " \
            "are {}.".format(self.__name__, self.layer_name, net.layers.keys())

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        log = OrderedDict()
        input_log = log['inputs'] = OrderedDict()
        for key, v in net.buffer[self.layer_name].inputs.items():
            v = net.handler.get_numpy_copy(v)
            key_log = input_log[key] = OrderedDict()
            key_log['min'] = v.min()
            key_log['avg'] = v.mean()
            key_log['max'] = v.max()

        output_log = log['outputs'] = OrderedDict()
        for key, v in net.buffer[self.layer_name].outputs.items():
            key_log = output_log[key] = OrderedDict()
            v = net.handler.get_numpy_copy(v)
            key_log['min'] = v.min()
            key_log['avg'] = v.mean()
            key_log['max'] = v.max()

        return log


class StopAfterEpoch(Hook):
    def __init__(self, max_epochs, timescale='epoch', interval=1, name=None,
                 verbose=None):
        super(StopAfterEpoch, self).__init__(name, timescale,
                                             interval, verbose)
        self.max_epochs = max_epochs

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        if epoch_nr >= self.max_epochs:
            raise StopIteration("Maximum number of epochs ({}) reached."
                                .format(self.max_epochs))


class EarlyStopper(Hook):
    __default_values__ = {'patience': 1}

    def __init__(self, log_name, patience=1, name=None):
        super(EarlyStopper, self).__init__(name, 'epoch', 1)
        self.log_name = log_name
        self.patience = patience

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        e = get_by_path(logs, self.log_name)
        best_error_idx = np.argmin(e)
        if len(e) > best_error_idx + self.patience:
            raise StopIteration("Error did not fall for %d epochs! Stopping."
                                % self.patience)


class StopOnNan(Hook):
    """ Stop the training if infinite or NaN values are found in parameters.

    Can also check logs for invalid values.
    """
    def __init__(self, logs_to_check=(), check_parameters=True,
                 check_training_loss=True, name=None, timescale='epoch',
                 interval=1):
        super(StopOnNan, self).__init__(name, timescale, interval)
        self.logs_to_check = ([logs_to_check] if isinstance(logs_to_check,
                                                            string_types)
                              else logs_to_check)
        self.check_parameters = check_parameters
        self.check_training_loss = check_training_loss

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        for log_name in self.logs_to_check:
            log = get_by_path(logs, log_name)
            if not np.all(np.isfinite(log)):
                raise StopIteration("{} >> NaN or inf detected in {}"
                                    .format(self.__name__, log_name))
        if self.check_parameters:
            params = net.handler.get_numpy_copy(net.buffer.parameters)
            if not np.all(np.isfinite(params)):
                raise StopIteration("{} >> NaN or inf detected in parameters!"
                                    .format(self.__name__))

        if self.check_training_loss and logs['training_loss']:
            if not np.all(np.isfinite(logs['training_loss'][1:])):
                raise StopIteration("{} >> NaN or inf detected in "
                                    "training_loss!".format(self.__name__))


class InfoUpdater(Hook):
    """ Save the information from logs to the Sacred custom info dict"""
    def __init__(self, run, name=None):
        super(InfoUpdater, self).__init__(name, 'epoch', 1)
        self.run = run
        self.__name__ = self.__class__.__name__ if name is None else name

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        info = self.run.info
        info['epoch'] = epoch_nr
        info['monitor'] = logs
        if 'nr_parameters' not in info:
            info['nr_parameters'] = net.buffer.parameters.size


class MonitorLoss(Hook):
    def __init__(self, iter_name, timescale='epoch', interval=1, name=None,
                 verbose=None):
        super(MonitorLoss, self).__init__(name, timescale, interval, verbose)
        self.iter_name = iter_name
        self.iter = None

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorLoss, self).start(net, stepper, verbose, monitor_kwargs)
        assert self.iter_name in monitor_kwargs
        self.iter = monitor_kwargs[self.iter_name]

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        iterator = self.iter(verbose=self.verbose, handler=net.handler)
        loss = []
        for _ in run_network(net, iterator):
            net.forward_pass()
            loss.append(net.get_loss_value())
        return np.mean(loss)


class MonitorAccuracy(Hook):
    """
    Monitor the classification accuracy of a given layer wrt. to given targets
    using a given data iterator.

    Parameters
    ----------
    iter_name : str
        name of the data iterator to use (as specified in the train() call)
    output : str
        name of the output to use formatted like this:
        LAYER_NAME[.OUTPUT_NAME]
        Where OUTPUT_NAME defaults to 'default'
    targets_name : str, optional
        name of the targets (as specified in the InputLayer)
        defaults to 'targets'


    Other Parameters
    ----------------
    timescale : {'epoch', 'update'}, optional
        Specifies whether the Monitor should be called after each epoch or
        after each update. Default is 'epoch'
    interval : int, optional
        This monitor should be called every ``interval`` epochs/updates.
        Default is 1
    name: str, optional
        Name of this monitor. This name is used as a key in the trainer logs.
        Default is 'MonitorAccuracy'
    verbose: bool, optional
        Specifies whether the logs of this monitor should be printed, and
        acts as a fallback verbosity for the used data iterator.
        If not set it defaults to the verbosity setting of the trainer.

    See Also
    --------
    MonitorLoss : monitor the overall loss of the network.
    MonitorHammingScore : monitor the Hamming score which is a generalization
        of accuracy for multi-label classification tasks.

    Notes
    -----
    Can be used both with integer and one-hot targets.

    """
    def __init__(self, iter_name, output, targets_name='targets',
                 mask_name='mask', timescale='epoch', interval=1,
                 name=None, verbose=None):

        super(MonitorAccuracy, self).__init__(name, timescale, interval,
                                              verbose)
        self.iter_name = iter_name
        self.out_layer, _, self.out_name = output.partition('.')
        self.out_name = self.out_name or 'default'
        self.targets_name = targets_name
        self.mask_name = mask_name
        self.iter = None
        self.masked = False

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorAccuracy, self).start(net, stepper, verbose,
                                           monitor_kwargs)
        assert self.iter_name in monitor_kwargs, \
            "{} >> {} is not present in monitor_kwargs. Remember to pass it " \
            "as kwarg to Trainer.train().".format(self.__name__,
                                                  self.iter_name)
        assert self.out_layer in net.layers
        self.iter = monitor_kwargs[self.iter_name]
        self.masked = self.mask_name in self.iter.data.keys()

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        iterator = self.iter(verbose=self.verbose, handler=net.handler)
        _h = net.handler
        errors = 0
        totals = 0
        loss = []
        log = OrderedDict()
        for _ in run_network(net, iterator):
            net.forward_pass()
            loss.append(net.get_loss_value())
            out = _h.get_numpy_copy(net.buffer[self.out_layer]
                                    .outputs[self.out_name])
            target = _h.get_numpy_copy(net.buffer.Input
                                       .outputs[self.targets_name])

            out = out.reshape(out.shape[0], out.shape[1], -1)
            target = target.reshape(target.shape[0], target.shape[1], -1)

            out_class = np.argmax(out, axis=2)
            if target.shape[2] > 1:
                target_class = np.argmax(target, axis=2)
            else:
                target_class = target[:, :, 0]

            assert out_class.shape == target_class.shape

            if self.masked:
                mask = _h.get_numpy_copy(net.buffer.Input
                                         .outputs[self.mask_name])[:, :, 0]
                errors += np.sum((out_class != target_class) * mask)
                totals += np.sum(mask)
            else:
                errors += np.sum(out_class != target_class)
                totals += np.prod(target_class.shape)

        log['accuracy'] = 1.0 - errors / totals
        log['loss'] = np.mean(loss)
        return log


class MonitorHammingScore(Hook):
    r"""
    Monitor the Hamming score of a given layer wrt. to given targets
    using a given data iterator.

    Hamming loss is defined as the fraction of the correct labels to the
    total number of labels.


    Parameters
    ----------
    iter_name : str
        name of the data iterator to use (as specified in the train() call)
    output : str
        name of the output to use formatted like this:
        LAYER_NAME[.OUTPUT_NAME]
        Where OUTPUT_NAME defaults to 'default'
    targets_name : str, optional
        name of the targets (as specified in the Input)
        defaults to 'targets'


    Other Parameters
    ----------------
    timescale : {'epoch', 'update'}, optional
        Specifies whether the Monitor should be called after each epoch or
        after each update. Default is 'epoch'
    interval : int, optional
        This monitor should be called every ``interval`` epochs/updates.
        Default is 1
    name: str, optional
        Name of this monitor. This name is used as a key in the trainer logs.
        Default is 'MonitorAccuracy'
    verbose: bool, optional
        Specifies whether the logs of this monitor should be printed and
        acts as a fallback verbosity for the used data iterator.
        If not set it defaults to the verbosity setting of the trainer.

    See Also
    --------
    MonitorLoss : monitor the overall loss of the network.
    MonitorAccuracy : monitor the classification accuracy
    """
    def __init__(self, iter_name, output, targets_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        super(MonitorHammingScore, self).__init__(name, timescale, interval,
                                                  verbose)
        self.iter_name = iter_name
        self.out_layer, _, self.out_name = output.partition('.')
        self.out_name = self.out_name or 'default'
        self.targets_name = targets_name
        self.iter = None

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorHammingScore, self).start(net, stepper, verbose,
                                               monitor_kwargs)
        assert self.iter_name in monitor_kwargs
        assert self.out_layer in net.layers
        self.iter = monitor_kwargs[self.iter_name]

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        iterator = self.iter(verbose=self.verbose, handler=net.handler)
        _h = net.handler
        errors = 0
        totals = 0
        for _ in run_network(net, iterator):
            net.forward_pass()
            out = _h.get_numpy_copy(net.buffer[self.out_layer]
                                    .outputs[self.out_name])
            target = _h.get_numpy_copy(net.buffer.Input
                                       .outputs[self.targets_name])

            out = out.reshape(out.shape[0], out.shape[1], -1)
            target = target.reshape(target.shape[0], target.shape[1], -1)

            errors += np.sum(np.logical_xor(out >= 0.5, target))
            totals += np.prod(target.shape)

        return 1.0 - errors / totals


class VisualiseAccuracy(Hook):
    """
    Visualises the accuracy using the bokeh.plotting library.

    By default the output saved as a .html file, however a display can be enabled

    Parameters
    ----------
    log_names : list, array, or dict
        Contains the name of the accuracies recorded by the accuracy monitors.
        Input should be of the form <monitorname>.accuracy
    filename : str
        The location to which the .html file containing the accuracy plot should be saved
    display : boolean
        If set to true bokeh will launch a tab in your default browser after EVERY timescale
        and display the intermediate accuracy plot
    """
    def __init__(self, log_names, filename, display=False, timescale='epoch', interval=1, name=None, verbose=None):
        super(VisualiseAccuracy, self).__init__(name, timescale, interval, verbose)

        self.log_names = log_names
        self.filename = filename

        try:
            import bokeh.plotting as bk

            self.bk = bk
            self.bk.output_file(self.filename + ".html", title="Accuracy Monitor", mode="cdn")
            self.TOOLS = "resize,crosshair,pan,wheel_zoom,box_zoom,reset"
            self.colors = ['blue', 'green', 'red', 'olive', 'cyan', 'aqua', 'gray']
            self.display = display

        except ImportError:
            print("bokeh is required for drawing networks but was not found.")

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):

        x_max = 0
        if self.timescale == 'epoch':
            x_max = epoch_nr + 3
        elif self.timescale == 'update':
            x_max = update_nr + 3

        fig = self.bk.figure(title="Accuracy Monitor", x_axis_label=self.timescale, y_axis_label='accuracy',
                             tools=self.TOOLS, x_range=(0, x_max), y_range=(0, 1))
        count = 0
        for log_name in self.log_names:
            e = get_by_path(logs, log_name)

            fig.line(range(len(e)), e, legend=log_name[0], line_width=2, color=self.colors[count])
            count += 1

        self.bk.save(fig, filename=self.filename + ".html")

        if self.display:
            self.bk.show(fig)
