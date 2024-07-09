
select  count(*)
from	
	web_sales
    	,item 
    	,date_dim
where 
	ws_item_sk = i_item_sk 
  	and i_category in ('Electronics', 'Children', 'Books')
  	and ws_sold_date_sk = d_date_sk
	and d_date between cast('1999-02-07' as date) 
				and (cast('1999-02-07' as date) + interval '30 day');


