"""
Item 45: Consider @property Instead of Refactoring Attributes
"""
def test_property():
    import pytest
    class People:
        def __init__(self, name: str, weight: float, height: float):
            self._name = name
            self._weight = weight
            self._height = height

        @property
        def bmi(self):
            return self._weight / (self._height ** 2)
        
        @property
        def weight(self):
            return self._weight
        
        @weight.setter
        def weight(self, value):
            if value < 0:
                raise ValueError('Weight must be non-negative')
            self._weight = value
        
        @property
        def height(self):
            return self._height
        
        @height.setter
        def height(self, value):
            if value < 0:
                raise ValueError('Height must be non-negative')
            self._height = value

    p = People('John', 70, 1.75)
    assert p.bmi == pytest.approx(22.86, 0.1)
    assert p.weight == 70
    assert p.height == 1.75

    with pytest.raises(ValueError):
        p.weight = -1