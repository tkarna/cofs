"""
Utility function and extensions to traitlets used for specifying Thetis options
"""
from ipython_genutils.text import indent, dedent
from traitlets.config.configurable import Configurable
from traitlets import *
from firedrake import Constant, Function

import ufl

from abc import ABCMeta, abstractproperty


def rst_all_options(cls, nspace=0, prefix=None):
    """Recursively generate rST for a provided Configurable class.

    :arg cls: The Configurable class.
    :arg nspace: Indentation level.
    :arg prefix: Prefix to use for new traits."""
    lines = []
    classname = cls.__name__

    # Slaved options don't appear directly, but underneath their controlling enum.
    slaved_options = set()
    for trait in cls.class_own_traits().values():
        if isinstance(trait, PairedEnum):
            slaved_options.add(trait.paired_name)
    for k, trait in sorted(cls.class_own_traits(config=True).items()):
        typ = trait.__class__.__name__
        if trait.name in slaved_options:
            continue
        _prefix = prefix if prefix is not None else classname
        termline = r"{prefix:}.\ **{suffix:}**".format(prefix=_prefix, suffix=trait.name)

        if 'Enum' in typ:
            termline += ' : ' + '|'.join(repr(x) for x in trait.values)
        else:
            termline += ' : ' + typ
        lines.append(indent(termline, nspace))

        if isinstance(trait, PairedEnum):
            # Find the slaved option and recurse to fill in the subtree.
            dvr = trait.default_value_repr()
            extra = ["",
                     "Setting value implies configuration of sub-tree %s.%s:" % (classname, trait.paired_name),
                     ""]
            for opt, val in trait.paired_defaults.items():
                extra.append("'%s':" % opt)
                extra.append("")
                extra.append(rst_all_options(val, 4 + nspace, prefix=classname + "." + trait.paired_name))
            extra = "\n".join(extra)
        else:
            extra = None
            try:
                dvr = trait.default_value_repr()
            except Exception:
                dvr = None
        help = trait.help.strip() if trait.help else 'No description'
        lines.append(indent(dedent(help), 4 + nspace))
        lines.append('')
        lines.append(indent("Default:\n", 4 + nspace))
        dvr = dvr or 'None'
        lines.append(indent(dvr.replace("\\n", "\\\\n"), 4 + nspace))
        if extra is not None:
            lines.append(indent(extra, 4 + nspace))
        lines.append('')
    return "\n".join(lines)


class PositiveInteger(Integer):
    def info(self):
        return u'a positive integer'

    def validate(self, obj, proposal):
        super(PositiveInteger, self).validate(obj, proposal)
        assert proposal > 0, self.error(obj, proposal)
        return proposal


class PositiveFloat(Float):
    def info(self):
        return u'a positive float'

    def validate(self, obj, proposal):
        super(PositiveFloat, self).validate(obj, proposal)
        assert proposal > 0.0, self.error(obj, proposal)
        return proposal


class NonNegativeInteger(Integer):
    def info(self):
        return u'a non-negative integer'

    def validate(self, obj, proposal):
        super(NonNegativeInteger, self).validate(obj, proposal)
        assert proposal >= 0, self.error(obj, proposal)
        return proposal


class NonNegativeFloat(Float):
    def info(self):
        return u'a non-negative float'

    def validate(self, obj, proposal):
        super(NonNegativeFloat, self).validate(obj, proposal)
        assert proposal >= 0.0, self.error(obj, proposal)
        return proposal


class BoundedInteger(Integer):
    def __init__(self, default_value=Undefined, bounds=None, **kwargs):
        super(BoundedInteger, self).__init__(default_value, **kwargs)
        self.minval = bounds[0]
        self.maxval = bounds[1]

    def info(self):
        return u'an integer between {:} and {:}'.format(self.minval, self.maxval)

    def validate(self, obj, proposal):
        super(BoundedInteger, self).validate(obj, proposal)
        assert proposal >= self.minval, self.error(obj, proposal)
        assert proposal <= self.maxval, self.error(obj, proposal)
        return proposal


class BoundedFloat(Float):
    def __init__(self, default_value=Undefined, bounds=None, **kwargs):
        self.minval = bounds[0]
        self.maxval = bounds[1]
        super(BoundedFloat, self).__init__(default_value, **kwargs)

    def info(self):
        return u'a float between {:} and {:}'.format(self.minval, self.maxval)

    def validate(self, obj, proposal):
        super(BoundedFloat, self).validate(obj, proposal)
        assert proposal >= self.minval, self.error(obj, proposal)
        assert proposal <= self.maxval, self.error(obj, proposal)
        return proposal


class FiredrakeConstantTraitlet(TraitType):
    default_value = None
    info_text = 'a Firedrake Constant'

    def validate(self, obj, value):
        if isinstance(value, Constant):
            return value
        self.error(obj, value)

    def default_value_repr(self):
        return 'Constant({:})'.format(self.default_value.dat.data[0])


class FiredrakeCoefficient(TraitType):
    default_value = None
    info_text = 'a Firedrake Constant or Function'

    def validate(self, obj, value):
        if isinstance(value, (Constant, Function)):
            return value
        self.error(obj, value)

    def default_value_repr(self):
        if isinstance(self.default_value, Constant):
            return 'Constant({:})'.format(self.default_value.dat.data[0])
        return 'Function'


class FiredrakeScalarExpression(TraitType):
    default_value = None
    info_text = 'a scalar UFL expression'

    def validate(self, obj, value):
        if (isinstance(value, ufl.core.expr.Expr)
                and ufl.checks.is_ufl_scalar(value)):
            return value
        self.error(obj, value)

    def default_value_repr(self):
        if isinstance(self.default_value, Constant):
            return 'Constant({:})'.format(self.default_value.dat.data[0])
        if isinstance(self.default_value, Function):
            return 'Function'
        return 'UFL scalar expression'


class FiredrakeVectorExpression(TraitType):
    default_value = None
    info_text = 'a vector UFL expression'

    def validate(self, obj, value):
        if (isinstance(value, ufl.core.expr.Expr)
                and not ufl.checks.is_ufl_scalar(value)):
            return value
        self.error(obj, value)

    def default_value_repr(self):
        if isinstance(self.default_value, Constant):
            return 'Constant({:})'.format(self.default_value.dat.data[0])
        if isinstance(self.default_value, Function):
            return 'Function'
        return 'UFL vector expression'


class PETScSolverParameters(Dict):
    """PETSc solver options dictionary"""
    info_text = 'a PETSc solver options dictionary'

    def validate(self, obj, value):
        if isinstance(value, dict):
            return value
        self.error(obj, value)


class PairedEnum(Enum):
    """A enum whose value must be in a given sequence.

    This enum controls a slaved option, with default values provided here.

    :arg values: iterable of (value, HasTraits class) pairs.
        The HasTraits class will be called (with no arguments) to
        create default values if necessary.
    :arg paired_name: trait name this enum is paired with.
    :arg default_value: default value.
    """
    def __init__(self, values, paired_name, default_value=Undefined, **kwargs):
        self.paired_defaults = dict(values)
        self.paired_name = paired_name
        values, _ = zip(*values)
        super(PairedEnum, self).__init__(values, default_value, **kwargs)

    def info(self):
        result = "This option also requires configuration of %s\n" % (self.paired_name)
        return result + super(PairedEnum, self).info()


class OptionsBase(object):
    """Abstract base class for all options classes"""

    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        """Human readable name of the configurable object"""
        pass

    def update(self, options):
        """
        Assign options from another container

        :arg options: Either a dictionary of options or another
            HasTraits object
        """
        if isinstance(options, dict):
            params_dict = options
        else:
            assert isinstance(options, HasTraits), 'options must be a dict or HasTraits object'
            params_dict = options._trait_values
        for key in params_dict:
            self.__setattr__(key, params_dict[key])

    def __str__(self):
        """Returs a summary of all defined parameters and their values in a string"""
        output = '{:} parameters\n'.format(self.name)
        params_dict = self._trait_values
        for k in sorted(params_dict.keys()):
            output += '  {:16s} : {:}\n'.format(k, params_dict[k])
        return output


# HasTraits and Configurable (and all their subclasses) have MetaHasTraits as their metaclass
# to subclass from HasTraits and another class with ABCMeta as its metaclass, we need a combined
# meta class that sub(meta)classes from ABCMeta and MetaHasTraits
class ABCMetaHasTraits(ABCMeta, MetaHasTraits):
    """Combined metaclass of ABCMeta and MetaHasTraits"""
    pass


class FrozenHasTraits(OptionsBase, HasTraits):
    __metaclass__ = ABCMetaHasTraits
    """
    A HasTraits class that only allows adding new attributes in the class
    definition or when  self._isfrozen is False.
    """
    _isfrozen = False

    def __init__(self, *args, **kwargs):
        super(FrozenHasTraits, self).__init__(*args, **kwargs)
        self._isfrozen = True

    def __setattr__(self, key, value):
        if self._isfrozen and not hasattr(self, key):
            raise TypeError('Adding new attribute "{:}" to {:} class is forbidden'.format(key, self.__class__.__name__))
        super(FrozenHasTraits, self).__setattr__(key, value)


class FrozenConfigurable(OptionsBase, Configurable):
    """
    A Configurable class that only allows adding new attributes in the class
    definition or when  self._isfrozen is False.
    """
    __metaclass__ = ABCMetaHasTraits

    _isfrozen = False

    def __init__(self, *args, **kwargs):
        super(FrozenConfigurable, self).__init__(*args, **kwargs)
        self._isfrozen = True

    def __setattr__(self, key, value):
        if self._isfrozen and not hasattr(self, key):
            raise TypeError('Adding new attribute "{:}" to {:} class is forbidden'.format(key, self.__class__.__name__))
        super(FrozenConfigurable, self).__setattr__(key, value)


def attach_paired_options(name, name_trait, value_trait):
    """Attach paired options to a Configurable object.

    :arg name: the name of the enum trait
    :arg name_trait: the enum trait (a PairedEnum)
    :arg value_trait: the slaved value trait."""

    def _observer(self, change):
        "Observer called when the choice option is updated."
        setattr(self, name_trait.paired_name,
                name_trait.paired_defaults[change["new"]]())

    def _default(self):
        "Dynamic default value setter"
        if hasattr(name_trait, 'default_value') and name_trait.default_value is not None:
            return name_trait.paired_defaults[name_trait.default_value]()

    obs_handler = observe(name, type="change")
    def_handler = default(name_trait.paired_name)

    def update_class(cls):
        "Programmatically update the class"
        # Set the new class attributes
        setattr(cls, name, name_trait)
        setattr(cls, name_trait.paired_name, value_trait)
        setattr(cls, "_%s_observer" % name, obs_handler(_observer))
        setattr(cls, "_%s_default" % name_trait.paired_name, def_handler(_default))
        # Mimic the magic metaclass voodoo.
        name_trait.class_init(cls, name)
        value_trait.class_init(cls, name_trait.paired_name)
        obs_handler.class_init(cls, "_%s_observer" % name)
        def_handler.class_init(cls, "_%s_default" % name_trait.paired_name)

        return cls
    return update_class
