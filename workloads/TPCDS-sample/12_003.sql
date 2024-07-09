
select  count(*)
from	
	web_sales
    	,item 
    	,date_dim
where 
	ws_item_sk = i_item_sk 
  	and i_category in ('Home', 'Children', 'Shoes')
  	and ws_sold_date_sk = d_date_sk
	and d_date between cast('2002-04-18' as date) 
				and (cast('2002-04-18' as date) + interval '30 day');


