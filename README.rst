|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Дистилляция знаний в глубоких сетях с применением методов выравнивания структур моделей
    :Тип научной работы: НИР
    :Автор: Михаил Сергеевич Олейник
    :Научный руководитель: кандидат физико-математических наук, Бахтеев Олег Юрьевич

Abstract
========

Дистилляция знаний позволяет повысить качество модели, называемой учеником, не увеличивая её число параметров,
а используя модель большего размера, называемой учителем.
Однако, в случае разных архитектур и несовпадения количества слоев у учителя и ученика, распространённые методы не применимы.
Одним из подходов, который позволяет решать задачу для разных архитектур, является максимизация взаимной информации.
Мы предлагаем улучшение этого подхода, которое позволит проводить дистилляцию и для моделей с разным количеством слоёв.

Software modules developed as part of the study
======================================================
1. A python code with all implementation `here <https://github.com/intsystems/Oleinik-BS-Thesis/blob/master/code>`_.
2. A code with base experiment `here <https://github.com/intsystems/Oleinik-BS-Thesis/blob/master/code/basic_experiment.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/Oleinik-BS-Thesis/blob/master/code/basic_experiment.ipynb>`_.
