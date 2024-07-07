
select  count(*)
from	
	web_sales
    	,item 
    	,date_dim
where 
	ws_item_sk = i_item_sk 
  	and i_category in ('Shoes', 'Men', 'Electronics')
  	and ws_sold_date_sk = d_date_sk
	and d_date between cast('2002-06-27' as date) 
				and (cast('2002-06-27' as date) + 30 days);


