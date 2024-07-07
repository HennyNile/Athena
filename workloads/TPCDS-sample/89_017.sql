
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2001) and
        ((i_category in ('Home','Jewelry','Sports') and
          i_class in ('paint','gold','athletic shoes')
         )
      or (i_category in ('Shoes','Music','Women') and
          i_class in ('kids','country','maternity') 
        ));


