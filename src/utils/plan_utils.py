import json

SCAN_TYPES = ['Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]

class UniquePlan:
    def __init__(self, plan: map):
        self.plan = plan
        self.signature, self.num_tables = self._signature(plan)

    def __eq__(self, other):
        return self.signature == other.signature
    
    def __hash__(self):
        return hash(tuple(self.signature))
    
    def __str__(self):
        str = ''
        def recurse(node: map, is_last: list[bool]):
            nonlocal str
            indent = ''
            if len(is_last) > 0:
                for last in is_last[:-1]:
                    indent += '  ' if last else '│ '
                indent += '└─' if is_last[-1] else '├─'
            node_type = node['Node Type']
            if node_type not in SCAN_TYPES + JOIN_TYPES:
                assert('Plans' not in node or len(node['Plans']) == 1)
                if 'Plans' in node:
                    recurse(node['Plans'][0], is_last)
            else:
                if node_type in SCAN_TYPES:
                    str += f'{indent}{node_type} on {node["Alias"]}\n'
                else:
                    str += f'{indent}{node_type}\n'
                for i, child in enumerate(node.get('Plans', [])):
                    recurse(child, is_last + [i == len(node['Plans']) - 1])
        recurse(self.plan, [])
        return str

    @staticmethod
    def _signature(plan: map) -> list:
        ret = []
        num_tables = 0

        def recurse(node: map):
            nonlocal num_tables
            ret.append('begin')
            node_type = node['Node Type']
            ret.append(node_type)
            if node_type in SCAN_TYPES:
                ret.append(node['Alias'])
                num_tables += 1
            for child in node.get('Plans', []):
                recurse(child)
            ret.append('end')
        
        recurse(plan)
        return ret, num_tables
    
def plan_struct_to_hint(tmp_sub_plan_struc, root=True):
    join_hints = [] # list of [join operator, tables]
    if len(tmp_sub_plan_struc) == 3:
        join_op = tmp_sub_plan_struc[0]
        left_tables, left_join_hints, left_leading_hint_text = plan_struct_to_hint(tmp_sub_plan_struc[1], root=False)
        right_tables, right_join_hints, right_leading_hint_text = plan_struct_to_hint(tmp_sub_plan_struc[2], root=False)
        join_tables = left_tables + right_tables
        curr_join_hint = join_op + "(" + " ".join(left_tables + right_tables) + ")"
        join_hints = [curr_join_hint] + left_join_hints + right_join_hints
        leading_hint_text = '({} {})'.format(left_leading_hint_text, right_leading_hint_text)

        if not root:
            return join_tables, join_hints, leading_hint_text
        else:
            join_hints_text = '/*+ ' + ' '.join(join_hints) + ' ' + 'Leading({})'.format(leading_hint_text) + ' */'
            return join_hints_text
    else:
        return tmp_sub_plan_struc, join_hints, tmp_sub_plan_struc[0]

def plan_struct_to_leading_hint(tmp_sub_plan_struc, root=True):
    join_hints = [] # list of [join operator, tables]
    if len(tmp_sub_plan_struc) == 3:
        join_op = tmp_sub_plan_struc[0]
        left_tables, left_join_hints, left_leading_hint_text = plan_struct_to_hint(tmp_sub_plan_struc[1], root=False)
        right_tables, right_join_hints, right_leading_hint_text = plan_struct_to_hint(tmp_sub_plan_struc[2], root=False)
        join_tables = left_tables + right_tables
        curr_join_hint = join_op + "(" + " ".join(left_tables + right_tables) + ")"
        join_hints = [curr_join_hint] + left_join_hints + right_join_hints
        leading_hint_text = '({} {})'.format(left_leading_hint_text, right_leading_hint_text)

        if not root:
            return join_tables, join_hints, leading_hint_text
        else:
            return leading_hint_text
    else:
        return tmp_sub_plan_struc, join_hints, tmp_sub_plan_struc[0]
    
def json_plan_to_plan_struct(tmp_json_plan):
    if tmp_json_plan['Node Type'] not in ['Hash Join', 'Nested Loop', 'Merge Join', 'Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan']:
        # print(tmp_json_plan['Node Type'])
        return json_plan_to_plan_struct(tmp_json_plan['Plans'][0])
    elif tmp_json_plan['Node Type'] in ['Hash Join', 'Nested Loop', 'Merge Join']:
        return [tmp_json_plan['Node Type'], json_plan_to_plan_struct(tmp_json_plan['Plans'][0]), json_plan_to_plan_struct(tmp_json_plan['Plans'][1])]
    else:
        return [tmp_json_plan['Alias']]    

def plan_to_leading_hint(json_plan: dict):
    json_struct = json_plan_to_plan_struct(json_plan['Plan'])
    leading_hint_text = plan_struct_to_leading_hint(json_struct)
    return leading_hint_text