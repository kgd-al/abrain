.. note:: Automatically extracted from sources on Sun Jan  1 12:20:42 CET 2023 for version 0.1b

Identity
________

.. grid:: auto
  :gutter: 0

  .. grid-item-card::
    :columns: 2
    :text-align: right

    .. image:: ../../../src/abrain/core/functions/id.svg

  .. grid-item-card::
    :columns: 10

    .. math:: x

  .. grid-item-card::
    :columns: 2
    :text-align: right

    id

  .. grid-item-card::
     :columns: 10

     .. plot::
       :height: 10em

       plt.plot(x, [functions['id'](x_) for x_ in x])

Absolute value
______________

.. grid:: auto
  :gutter: 0

  .. grid-item-card::
    :columns: 2
    :text-align: right

    .. image:: ../../../src/abrain/core/functions/abs.svg

  .. grid-item-card::
    :columns: 10

    .. math:: |x|

  .. grid-item-card::
    :columns: 2
    :text-align: right

    abs

  .. grid-item-card::
     :columns: 10

     .. plot::
       :height: 10em

       plt.plot(x, [functions['abs'](x_) for x_ in x])

Sinusoidal
__________

.. grid:: auto
  :gutter: 0

  .. grid-item-card::
    :columns: 2
    :text-align: right

    .. image:: ../../../src/abrain/core/functions/sin.svg

  .. grid-item-card::
    :columns: 10

    .. math:: sin(2x)

  .. grid-item-card::
    :columns: 2
    :text-align: right

    sin

  .. grid-item-card::
     :columns: 10

     .. plot::
       :height: 10em

       plt.plot(x, [functions['sin'](x_) for x_ in x])

Step function
_____________

.. grid:: auto
  :gutter: 0

  .. grid-item-card::
    :columns: 2
    :text-align: right

    .. image:: ../../../src/abrain/core/functions/step.svg

  .. grid-item-card::
    :columns: 10

    .. math:: 0 &\ \text{if } x \leq 0\\1 &\ \text{otherwise}

  .. grid-item-card::
    :columns: 2
    :text-align: right

    step

  .. grid-item-card::
     :columns: 10

     .. plot::
       :height: 10em

       plt.plot(x, [functions['step'](x_) for x_ in x])

Gaussian function
_________________

.. grid:: auto
  :gutter: 0

  .. grid-item-card::
    :columns: 2
    :text-align: right

    .. image:: ../../../src/abrain/core/functions/gaus.svg

  .. grid-item-card::
    :columns: 10

    .. math:: e^{-6.25x^2}

  .. grid-item-card::
    :columns: 2
    :text-align: right

    gaus

  .. grid-item-card::
     :columns: 10

     .. plot::
       :height: 10em

       plt.plot(x, [functions['gaus'](x_) for x_ in x])

Soft sigmoid
____________

.. grid:: auto
  :gutter: 0

  .. grid-item-card::
    :columns: 2
    :text-align: right

    .. image:: ../../../src/abrain/core/functions/ssgm.svg

  .. grid-item-card::
    :columns: 10

    .. math:: \frac{1}{1+e^{-4.9x}}

  .. grid-item-card::
    :columns: 2
    :text-align: right

    ssgm

  .. grid-item-card::
     :columns: 10

     .. plot::
       :height: 10em

       plt.plot(x, [functions['ssgm'](x_) for x_ in x])

Bimodal sigmoid
_______________

.. grid:: auto
  :gutter: 0

  .. grid-item-card::
    :columns: 2
    :text-align: right

    .. image:: ../../../src/abrain/core/functions/bsgm.svg

  .. grid-item-card::
    :columns: 10

    .. math:: \frac{2}{1+e^{-4.9x}} - 1

  .. grid-item-card::
    :columns: 2
    :text-align: right

    bsgm

  .. grid-item-card::
     :columns: 10

     .. plot::
       :height: 10em

       plt.plot(x, [functions['bsgm'](x_) for x_ in x])

Activation function
___________________

.. grid:: auto
  :gutter: 0

  .. grid-item-card::
    :columns: 2
    :text-align: right

    .. image:: ../../../src/abrain/core/functions/ssgn.svg

  .. grid-item-card::
    :columns: 10

    .. math:: e^{-(x+1)^2} - 1 &\ \text{if } x \lt -1 \\1 - e^{-(x-1)^2} &\ \text{if } x \gt  1 \\0 &\ \text{otherwise}

  .. grid-item-card::
    :columns: 2
    :text-align: right

    ssgn

  .. grid-item-card::
     :columns: 10

     .. plot::
       :height: 10em

       plt.plot(x, [functions['ssgn'](x_) for x_ in x])

