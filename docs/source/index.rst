.. IRM2 documentation master file

IRM2 — Neural Short-Rate Calibration Framework
==============================================

IRM2 is a **risk-neutral neural short-rate calibration framework** for U.S.
Treasury yield curves and Treasury futures.

The model encodes a history of yield curves into a latent state, evolves the
state under a learnable Neural SDE, decodes it into a short rate, and prices
both the yield curve and Treasury futures (via cheapest-to-deliver Monte
Carlo). Yields and futures are fit jointly under the risk-neutral measure
:math:`\mathbb{Q}`, with optional physical-measure (:math:`\mathbb{P}`)
consistency that identifies the market price of risk.

.. math::

   \mathcal{L}
   =
   \lambda_y \,\mathcal{L}_{\text{yield}}
   + \lambda_f \,\mathcal{L}_{\text{fut}}
   + \lambda_c \,\mathcal{L}_{\text{cons}}
   + \lambda_\sigma \,\mathcal{L}_{\text{vol}}
   + \lambda_{\mathbb{P}} \,\mathcal{L}_{\mathbb{P}/\mathbb{Q}} .

See :doc:`concepts/formulation` for the full mathematical specification, and
the ``report/`` folder (PDF/HTML) for the modelling diagnostics
behind each term.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   overview
   install
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/formulation
   concepts/dataflow
   concepts/types
   concepts/pricing
   concepts/training

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/01_load_market_data
   examples/02_build_and_train_model
   examples/03_evaluate_and_inspect
   examples/04_futures_pricing_walkthrough
   examples/05_gridsearch

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/configs
   api/dataloaders
   api/models
   api/finance
   api/training
   api/types

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
