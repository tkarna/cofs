
======================
 Thetis documentation
======================

Installation
============

See :doc:`obtaining Thetis <download>` for installation instructions.

Tutorials
=========

Once Thetis is successfully installed, you can start running example
simulations that demonstrate basic functionality.

.. note::

    Setting up Thetis simulations makes use of Firedrake objects,
    such as meshes, functions, and expression.
    In order to be able to follow these demo simulations,
    it is essential to have a basic understanding of these objects.
    Please refer to the
    `Firedrake manual <http://firedrakeproject.org/documentation.html>`_
    for more information.
    `Defining variational problems <http://firedrakeproject.org/variational-problems.html>`_
    page is a good primer for understanding the Firedrake concepts.


.. toctree::
    :maxdepth: 1

    2D channel with closed boundaries<demos/demo_2d_channel.py>
    2D channel with boundary conditions<demos/demo_2d_channel_bnd.py>
    3D channel with boundary conditions<demos/demo_3d_channel.py>
    2D tracer advection in a rotational velocity field <demos/demo_2d_tracer.py>

Manual
======

.. toctree::
    :maxdepth: 1

    2D hydrodynamics model formulation<model_formulation_2d>
    2D tracer model formulation<tracer_formulation_2d>
    2D sediment model formulation<sediment_formulation_2d>
    3D model formulation<model_formulation_3d>
    Model outputs and visualization<outputs_and_visu>
    List of 2D model options<model_options_2d>
    List of 3D model options<model_options_3d>
    List of physical fields<field_documentation>

API documentation
=================

The complete list of all the classes and methods is
available at the :doc:`thetis` page. The same information is :ref:`indexed
<genindex>` in alphabetical order. Another very effective mechanism is
the site :ref:`search engine <search>`.
