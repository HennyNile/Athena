SCAN_TYPES = ['Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan']

class UniquePlan:
    def __init__(self, plan: map):
        self.plan = plan
        self.signature, self.num_tables = self._signature(plan)

    def __eq__(self, other):
        return self.signature == other.signature
    
    def __hash__(self):
        return hash(tuple(self.signature))

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