from packaging import version

try:
    import pydantic

    PYDANTIC_VERSION = version.parse(pydantic.__version__)
except ImportError:
    PYDANTIC_VERSION = version.parse('0.0.0')

IS_PYDANTIC_V1 = PYDANTIC_VERSION.major == 1
IS_PYDANTIC_V2 = PYDANTIC_VERSION.major == 2


def is_pydantic_instance(obj) -> bool:
    """
    Проверяет, является ли переданный объект экземпляром Pydantic модели.

    Поддерживает:
        - pydantic.BaseModel в Pydantic 1.x
        - pydantic.BaseModel в Pydantic 2.x
        - pydantic.v1.BaseModel в Pydantic 2.x

    Args:
        obj: Объект для проверки

    Returns:
        bool: True, если объект является экземпляром Pydantic модели, иначе False
    """
    # Если объект None или не является объектом класса, возвращаем False
    if obj is None:
        return False

    if IS_PYDANTIC_V1:
        from pydantic import BaseModel as BaseModelV1

        return isinstance(obj, BaseModelV1)

    elif IS_PYDANTIC_V2:
        from pydantic import BaseModel as BaseModelV2
        from pydantic.v1 import BaseModel as BaseModelV1

        return isinstance(obj, (BaseModelV1, BaseModelV2))

    return False
