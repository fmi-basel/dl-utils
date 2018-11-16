from queue import Queue
from threading import Thread
import logging

# sentinel used as stop signal.
_SENTINEL = object()


def _preprocessor(out_queue, preprocessor_fn, vals):
    '''preprocessor that produces intermediate results
    for each value in vals.

    Parameters
    ----------
    out_queue : queue
        output queue onto which shall be produced.
    fn : function
        producing function, takes val in vals.
    vals : iterable
        iterable for input values to producer.

    '''
    logging.getLogger(__name__).debug('Preprocessor starting')
    debug = logging.getLogger(__name__).isEnabledFor(logging.DEBUG)
    for val in vals:
        if debug:
            logging.getLogger(__name__).debug('Preprocessor handling val: %s',
                                              str(val))
        out_queue.put(preprocessor_fn(val))

    # mark "end" for consumers.
    out_queue.put(_SENTINEL)
    logging.getLogger(__name__).debug('Preprocessor exiting')


def _postprocessor(in_queue, postprocessor_fn):
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
            vals = tuple(vals)

        postprocessor_fn(*vals)
        in_queue.task_done()

    logging.getLogger(__name__).debug('Postprocessor exiting')


def _processor(in_queue, out_queue, processor_fn):
    '''
    '''
    logging.getLogger(__name__).debug('Processor starting')
    while True:
        vals = in_queue.get()

        if vals is _SENTINEL:
            in_queue.put(vals)
            in_queue.task_done()

            out_queue.put(vals)
            break

        if not isinstance(vals, tuple):
            vals = tuple(vals)

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

    '''
    assert queue_maxsize >= 1
    assert len(vals) >= 1

    logging.getLogger(__name__).debug('Runner starting')

    in_queue = Queue(maxsize=queue_maxsize)
    out_queue = Queue(maxsize=queue_maxsize)

    # setup threads
    preprocessor_thread = Thread(
        target=_preprocessor, args=(in_queue, preprocessor_fn, vals),
        daemon=True)

    postprocessor_thread = Thread(
        target=_postprocessor, args=(out_queue, postprocessor_fn),
        daemon=True)
    preprocessor_thread.start()
    postprocessor_thread.start()

    # processor runs in the main thread.
    _processor(in_queue, out_queue, processor_fn)

    # we are done once the postprocessor exits.
    postprocessor_thread.join()

    logging.getLogger(__name__).debug('Runner exiting')
