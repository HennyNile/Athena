


select
	count(*)
from
	customer,
	orders,
	lineitem
where
	c_mktsegment = 'MACHINERY'
	and c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and o_orderdate < date '1995-03-22'
	and l_shipdate > date '1995-03-22';
