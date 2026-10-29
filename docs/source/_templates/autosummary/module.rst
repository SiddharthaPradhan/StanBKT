{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/internal_function.rst

{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/internal_class.rst

{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/internal_class.rst

{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}

.. rubric:: API Details

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index: