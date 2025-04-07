from threading import Thread

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class WrapperThread(Thread):
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
