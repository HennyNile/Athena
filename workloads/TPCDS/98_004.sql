
select count(*)
from	
	store_sales
    	,item 
    	,date_dim
where 
	ss_item_sk = i_item_sk 
  	and i_category in ('Home', 'Children', 'Shoes')
  	and ss_sold_date_sk = d_date_sk
	and d_date between cast('2001-05-23' as date) 
				and (cast('2001-05-23' as date) + 30 days);


