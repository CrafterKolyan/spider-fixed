################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3

from nltk import word_tokenize

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    values = {}
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2 + 1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2 + 1:]
        values[key] = val

    tokens = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(tokens)):
        if tokens[i] in values:
            tokens[i] = values[tokens[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(tokens) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = tokens[eq_idx - 1]
        if pre_tok in prefix:
            tokens = tokens[:eq_idx - 1] + [pre_tok + "="] + tokens[eq_idx + 1:]

    return tokens


def scan_alias(tokens):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(tokens) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[tokens[idx + 1]] = tokens[idx - 1]
    return alias


def get_tables_with_alias(schema, tokens):
    tables = scan_alias(tokens)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(tokens, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = tokens[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx + 1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx + 1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(tokens, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(tokens)
    is_block = False
    is_distinct = False
    if tokens[idx] == '(':
        is_block = True
        idx += 1

    if tokens[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(tokens[idx])
        idx += 1
        assert idx < len_ and tokens[idx] == '('
        idx += 1
        if tokens[idx] == "distinct":
            idx += 1
            is_distinct = True
        idx, col_id = parse_col(tokens, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and tokens[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, is_distinct)

    if tokens[idx] == "distinct":
        idx += 1
        is_distinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(tokens, idx, tables_with_alias, schema, default_tables)

    if is_block:
        assert tokens[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, is_distinct)


def parse_val_unit(tokens, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(tokens)
    is_block = False
    if tokens[idx] == '(':
        is_block = True
        idx += 1

    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(tokens, idx, tables_with_alias, schema, default_tables)
    col_unit2 = None
    if idx < len_ and tokens[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(tokens[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(tokens, idx, tables_with_alias, schema, default_tables)

    if is_block:
        assert tokens[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(tokens, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(tokens)
    key = tables_with_alias[tokens[idx]]

    if idx + 1 < len_ and tokens[idx + 1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(tokens, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(tokens)

    is_block = False
    if tokens[idx] == '(':
        is_block = True
        idx += 1

    if tokens[idx] == 'select':
        idx, val = parse_sql(tokens, idx, tables_with_alias, schema)
    elif "\"" in tokens[idx]:  # token is a string value
        val = tokens[idx]
        idx += 1
    else:
        try:
            val = float(tokens[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and tokens[end_idx] != ',' and tokens[end_idx] != ')' \
                    and tokens[end_idx] != 'and' and tokens[end_idx] not in CLAUSE_KEYWORDS and tokens[
                end_idx] not in JOIN_KEYWORDS:
                end_idx += 1

            idx, val = parse_col_unit(tokens[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if is_block:
        assert tokens[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(tokens, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(tokens)
    conditions = []

    while idx < len_:
        idx, val_unit = parse_val_unit(tokens, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if tokens[idx] == 'not':
            not_op = True
            idx += 1

        assert idx < len_ and tokens[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, tokens[idx])
        op_id = WHERE_OPS.index(tokens[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, val1 = parse_value(tokens, idx, tables_with_alias, schema, default_tables)
            assert tokens[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(tokens, idx, tables_with_alias, schema, default_tables)
        else:  # normal case: single value
            idx, val1 = parse_value(tokens, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conditions.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (tokens[idx] in CLAUSE_KEYWORDS or tokens[idx] in (")", ";") or tokens[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and tokens[idx] in COND_OPS:
            conditions.append(tokens[idx])
            idx += 1  # skip and/or

    return idx, conditions


def parse_select(tokens, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(tokens)

    assert tokens[idx] == 'select', "'select' not found"
    idx += 1
    is_distinct = False
    if idx < len_ and tokens[idx] == 'distinct':
        idx += 1
        is_distinct = True
    val_units = []

    while idx < len_ and tokens[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if tokens[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(tokens[idx])
            idx += 1
        idx, val_unit = parse_val_unit(tokens, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and tokens[idx] == ',':
            idx += 1  # skip ','

    return idx, (is_distinct, val_units)


def parse_from(tokens, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in tokens[start_idx:], "'from' not found"

    len_ = len(tokens)
    idx = tokens.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conditions = []

    while idx < len_:
        is_block = False
        if tokens[idx] == '(':
            is_block = True
            idx += 1

        if tokens[idx] == 'select':
            idx, sql = parse_sql(tokens, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            if idx < len_ and tokens[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(tokens, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'], table_unit))
            default_tables.append(table_name)
        if idx < len_ and tokens[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(tokens, idx, tables_with_alias, schema, default_tables)
            if len(conditions) > 0:
                conditions.append('and')
            conditions.extend(this_conds)

        if is_block:
            assert tokens[idx] == ')'
            idx += 1
        if idx < len_ and (tokens[idx] in CLAUSE_KEYWORDS or tokens[idx] in (")", ";")):
            break

    return idx, table_units, conditions, default_tables


def parse_where(tokens, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(tokens)

    if idx >= len_ or tokens[idx] != 'where':
        return idx, []

    idx += 1
    idx, conditions = parse_condition(tokens, idx, tables_with_alias, schema, default_tables)
    return idx, conditions


def parse_group_by(tokens, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(tokens)
    col_units = []

    if idx >= len_ or tokens[idx] != 'group':
        return idx, col_units

    idx += 1
    assert tokens[idx] == 'by'
    idx += 1

    while idx < len_ and not (tokens[idx] in CLAUSE_KEYWORDS or tokens[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(tokens, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and tokens[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(tokens, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(tokens)
    val_units = []
    order_type = 'asc'  # default type is 'asc'

    if idx >= len_ or tokens[idx] != 'order':
        return idx, val_units

    idx += 1
    assert tokens[idx] == 'by'
    idx += 1

    while idx < len_ and not (tokens[idx] in CLAUSE_KEYWORDS or tokens[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(tokens, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and tokens[idx] in ORDER_OPS:
            order_type = tokens[idx]
            idx += 1
        if idx < len_ and tokens[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(tokens, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(tokens)

    if idx >= len_ or tokens[idx] != 'having':
        return idx, []

    idx += 1
    idx, conditions = parse_condition(tokens, idx, tables_with_alias, schema, default_tables)
    return idx, conditions


def parse_limit(tokens, start_idx):
    idx = start_idx
    len_ = len(tokens)

    if idx < len_ and tokens[idx] == 'limit':
        idx += 2
        return idx, int(tokens[idx - 1])

    return idx, None


def parse_sql(tokens, start_idx, tables_with_alias, schema):
    is_block = False  # indicate whether this is a block of sql/sub-sql
    len_ = len(tokens)
    idx = start_idx

    sql = {}
    if tokens[idx] == '(':
        is_block = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(tokens, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(tokens, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(tokens, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(tokens, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(tokens, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(tokens, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(tokens, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(tokens, idx)
    if is_block:
        assert tokens[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(tokens, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and tokens[idx] in SQL_OPS:
        sql_op = tokens[idx]
        idx += 1
        idx, IUE_sql = parse_sql(tokens, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    tokens = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, tokens)
    _, sql = parse_sql(tokens, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(tokens, start_idx):
    idx = start_idx
    while idx < len(tokens) and tokens[idx] == ";":
        idx += 1
    return idx
