
select  count(*)
from	
	web_sales
    	,item 
    	,date_dim
where 
	ws_item_sk = i_item_sk 
  	and i_category in ('Women', 'Electronics', 'Sports')
  	and ws_sold_date_sk = d_date_sk
	and d_date between cast('1998-03-06' as date) 
				and (cast('1998-03-06' as date) + 30 days);


