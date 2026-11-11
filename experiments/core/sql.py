"""Quote SQL identifiers and literals without engine or model dependencies."""


def literal(text):
    return "'" + text.replace("'", "''") + "'"


def identifier(text):
    return '"' + text.replace('"', '""') + '"'
