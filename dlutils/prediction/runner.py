from queue import Queue
from threading import Thread
from threading import Event
from inspect import isgeneratorfunction
import logging

# sentinel used as stop signal.
_SENTINEL = object()


class FailEvent(Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.who = None
        self.error = None

    def set_exception(self, who, error):
        '''
        '''
        self.set()
        self.who = who
        self.error = error


def _preprocessor(out_queue, fail_event, preprocessor_fn, vals):
    '''preprocessor that produces intermediate results
    for each value in vals.

    Parameters
    ----------
    out_queue : queue
        output queue onto which shall be produced.
    fail_event : FailEvent
        signals an error in the pipeline.
    fn : function
        producing function, takes val in vals.
    vals : iterable
        iterable for input values to producer.

    '''
    logging.getLogger(__name__).debug('Preprocessor starting')

    is_generator = isgeneratorfunction(preprocessor_fn)
    for val in vals:
        if not isinstance(val, tuple):
            val = (val, )

        try:
            if is_generator:
                for retval in preprocessor_fn(*val):
                    out_queue.put(retval)
            else:
                out_queue.put(preprocessor_fn(*val))
        except Exception as err:
            if not fail_event.is_set():
                logging.getLogger(__name__).error(
                    'Postprocessor encountered: %s', str(err), exc_info=True)
                fail_event.set_exception('Preprocessor', err)

    # mark "end" for consumers.
    out_queue.put(_SENTINEL)
    logging.getLogger(__name__).debug('Preprocessor exiting')


def _postprocessor(in_queue, fail_event, postprocessor_fn):
    '''
    '''
    logging.getLogger(__name__).debug('Postprocessor starting')
    while True:
        vals = in_queue.get()

        # check for end.
        if vals is _SENTINEL:
            in_queue.task_done()
            # put mark back in case there are other postprocessors
            in_queue.put(_SENTINEL)
            break

        if not isinstance(vals, tuple):
            vals = (vals, )

        try:
            postprocessor_fn(*vals)
        except Exception as err:
            if not fail_event.is_set():
                logging.getLogger(__name__).error(
                    'Postprocessor encountered: %s', str(err), exc_info=True)
                fail_event.set_exception('Postprocessor', err)

        in_queue.task_done()

    logging.getLogger(__name__).debug('Postprocessor exiting')


def _processor(in_queue, out_queue, fail_event, processor_fn):
    '''
    '''
    logging.getLogger(__name__).debug('Processor starting')

    is_generator = isgeneratorfunction(processor_fn)

    while not fail_event.is_set():
        vals = in_queue.get()

        if vals is _SENTINEL:
            in_queue.put(vals)
            in_queue.task_done()

            out_queue.put(vals)
            break

        if not isinstance(vals, tuple):
            vals = (vals, )

        if is_generator:
            for retval in processor_fn(*vals):
                out_queue.put(retval)
        else:
            out_queue.put(processor_fn(*vals))
        in_queue.task_done()

    logging.getLogger(__name__).debug('Processor exiting')


def runner(preprocessor_fn,
           processor_fn,
           postprocessor_fn,
           vals,
           queue_maxsize=1):
    '''runs preprocessor and postprocessor as individual threads to allow
    for concurrent processing.

    Parameters
    ----------
    preprocessor_fn : function
        function to be called in the preprocessor.
    processor_fn : function
        function to be called in the processor. This is typically the
        computationally heavy part.
    postprocessor_fn : function
        function to be called in the postprocessor.
    vals : iterable
        input values for the runner.
    queue_maxsize : int
        maximum queue size.

    Notes
    -----
    Since this implementation uses ```threading.Thread```s, pre- and
    postprocessor functions should be limited to I/O operations in order
    to be efficient.

    Exceptions in the pre- and postprocessor will be caught and logged, but
    not re-raised. The runner then terminates after processor_fn finishes
    the current iteration.

    '''
    assert queue_maxsize >= 1
    assert len(vals) >= 1

    logging.getLogger(__name__).debug('Runner starting')

    in_queue = Queue(maxsize=queue_maxsize)
    out_queue = Queue(maxsize=queue_maxsize)
    fail_event = FailEvent()

    # setup threads
    preprocessor_thread = Thread(
        target=_preprocessor,
        args=(in_queue, fail_event, preprocessor_fn, vals),
        daemon=True)

    postprocessor_thread = Thread(
        target=_postprocessor,
        args=(out_queue, fail_event, postprocessor_fn),
        daemon=True)
    preprocessor_thread.start()
    postprocessor_thread.start()

    # processor runs in the main thread.
    _processor(in_queue, out_queue, fail_event, processor_fn)

    # Re-raise error from pre- or postprocessor if necessary.
    if fail_event.is_set():
        raise fail_event.error

    for thread in [preprocessor_thread, postprocessor_thread]:
        thread.join()

    logging.getLogger(__name__).debug('Runner exiting')
