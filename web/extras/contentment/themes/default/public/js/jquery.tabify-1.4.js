(function($){$.fn.extend({tabify:function(){function getHref(a){hash=$(a).find('a').attr('href');if(hash)return hash.substring(0,hash.length-4);else return false}function setActive(a){$(a).addClass('active');if(getHref(a))$(getHref(a)).show();else return false;$(a).siblings('li').each(function(){$(this).removeClass('active');$(getHref(this)).hide()})}return this.each(function(){var a=this;$(this).find('li a').each(function(){$(this).attr('href',$(this).attr('href')+'-tab')});function handleHash(){if(location.hash)setActive($(a).find('a[href='+location.hash+']').parent())}if(location.hash)handleHash();setInterval(handleHash,100);$(this).find('li').each(function(){if($(this).hasClass('active'))$(getHref(this)).show();else $(getHref(this)).hide()})})}})})(jQuery);
$(function(){ $('menu.tabs').tabify() });