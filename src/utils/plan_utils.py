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
                ret.append(node['Relation Name'])
                num_tables += 1
            for child in node.get('Plans', []):
                recurse(child)
            ret.append('end')
        
        recurse(plan)
        return ret, num_tables