
select count(*)
from	
	store_sales
    	,item 
    	,date_dim
where 
	ss_item_sk = i_item_sk 
  	and i_category in ('Home', 'Women', 'Men')
  	and ss_sold_date_sk = d_date_sk
	and d_date between cast('2000-04-11' as date) 
				and (cast('2000-04-11' as date) + interval '30 day');


