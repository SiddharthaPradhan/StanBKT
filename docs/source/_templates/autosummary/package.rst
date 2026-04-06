{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

{% if subpackages %}
.. rubric:: Subpackages

.. autosummary::
   :toctree:
   :recursive:

{% for item in subpackages %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree: _autosummary

{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree: _autosummary

{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
   :toctree: _autosummary

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