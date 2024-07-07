
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (1999) and
        ((i_category in ('Home','Sports','Jewelry') and
          i_class in ('lighting','golf','earings')
         )
      or (i_category in ('Children','Women','Shoes') and
          i_class in ('newborn','fragrances','athletic') 
        ));


