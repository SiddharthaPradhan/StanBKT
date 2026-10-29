{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}
{%- if fullname in public_names %}
   :no-index:
{%- endif %}
