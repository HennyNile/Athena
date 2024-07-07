
select count(*)
from	
	store_sales
    	,item 
    	,date_dim
where 
	ss_item_sk = i_item_sk 
  	and i_category in ('Women', 'Books', 'Jewelry')
  	and ss_sold_date_sk = d_date_sk
	and d_date between cast('1998-02-26' as date) 
				and (cast('1998-02-26' as date) + 30 days);


