{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. This template intelligently chooses between autopydantic_model and autoclass
.. based on whether the class inherits from pydantic.BaseModel

{% set obj = get_class_obj(fullname) %}
{% set is_pydantic = is_pydantic_model(obj) %}

{% if is_pydantic %}
.. autopydantic_model:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :special-members: __call__
{% else %}
.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :special-members: __call__
{% endif %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') or item in ['__call__'] %}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

