import atexit
import threading

import yaml

__all__ = ['NoAliasDumper', 'WrapperThread', 'CachedProperty', 'threaded']


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class WrapperThread(threading.Thread):
    def __init__(self, func, args):
        super(WrapperThread, self).__init__()
        self.result = None
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


class CachedProperty(object):
    """
    A custom property named CachedProperty is defined to cache the results of a given function. The idea is to replace
    the property with the cached value after the property has been evaluated, thus saving time for subsequent access.

    The CachedProperty class is implemented as a subclass of the object class and uses the __get__ method to handle
    access to properties. When the property is first accessed, it calls the wrapper function (func) and stores the
    result in the object's __dict__. This way, the property will be replaced by the cached value and the wrapped
    function will not be called again for the same object.

    A property that is only computed once per instance and then replaces itself with an ordinary attribute.

    The implementation refers to https://github.com/pydanny/cached-property/blob/master/cached_property.py .
        Note that this implementation does NOT work in multi-thread or coroutine scenarios.

    Usage:
        class Foo(object):
            @CachedProperty
            def foo(self):
                return some_expensive_computation()

        obj = Foo()
        print(obj.foo)
    """

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.__doc__ = getattr(func, '__doc__', '')

    def __get__(self, obj, cls):
        if obj is None:
            return self
        val = self.func(obj)
        # Hack __dict__ of obj to inject the value
        # Note that this is only executed once
        obj.__dict__[self.func.__name__] = val
        return val


_active_threads = []
_active_threads_lock = threading.Lock()


@atexit.register
def _thread_check():
    """程序退出时检查并等待所有存活的线程完成"""
    with _active_threads_lock:
        current_threads = list(_active_threads)
        _active_threads.clear()  # 清空列表避免重复处理

    alive_threads = [t for t in current_threads if t.is_alive()]
    for thread in alive_threads:
        thread.join()
    print("All threads are terminated")


def threaded(func):
    """
    Multi-threads a target function by default and returns the thread or function result.

    This decorator provides flexible execution of the target function, either in a separate thread or synchronously.
    By default, the function runs in a thread, but this can be controlled via the 'threaded=False' keyword argument
    which is removed from kwargs before calling the function.

    Args:
        func (callable): The function to be potentially executed in a separate thread.

    Returns:
        (callable): A wrapper function that either returns a daemon thread or the direct function result.

    Examples:
        >>> @threaded
        ... def process_data(data):
        ...     return data
        >>>
        >>> thread = process_data(my_data)  # Runs in background thread
        >>> result = process_data(my_data, threaded=False)  # Runs synchronously, returns function result
    """

    def wrapper(*args, **kwargs):
        """Multi-threads a given function based on 'threaded' kwarg and returns the thread or function result."""
        if kwargs.pop("threaded", True):  # run in thread
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            with _active_threads_lock:
                _active_threads.append(thread)
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper
