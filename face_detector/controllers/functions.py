# -*- coding: utf-8 -*-

from django.urls import reverse
import urllib.parse as urllib
from django.http import HttpResponseRedirect

def custom_redirect(url_name, *args, **kwargs):
    url = reverse(url_name, args = args)
    params = urllib.urlencode(kwargs)
    return HttpResponseRedirect(url + "?%s" % params)