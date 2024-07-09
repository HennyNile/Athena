
select count(*)
from	
	store_sales
    	,item 
    	,date_dim
where 
	ss_item_sk = i_item_sk 
  	and i_category in ('Jewelry', 'Music', 'Electronics')
  	and ss_sold_date_sk = d_date_sk
	and d_date between cast('1999-02-10' as date) 
				and (cast('1999-02-10' as date) + interval '30 day');


