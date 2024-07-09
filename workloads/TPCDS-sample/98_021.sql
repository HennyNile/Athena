
select count(*)
from	
	store_sales
    	,item 
    	,date_dim
where 
	ss_item_sk = i_item_sk 
  	and i_category in ('Children', 'Women', 'Shoes')
  	and ss_sold_date_sk = d_date_sk
	and d_date between cast('2000-01-21' as date) 
				and (cast('2000-01-21' as date) + interval '30 day');


