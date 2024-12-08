import rdflib
from rdflib import URIRef

WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')

class NamespaceWrapper:
    WD = WD
    WDT = WDT
    DDIS = DDIS
    RDFS = RDFS
    SCHEMA = SCHEMA

def uri_to_prefixed(uri):
    uri_ref = URIRef(uri)
    if str(uri_ref).startswith(str(WD)):
        return getattr(NamespaceWrapper, "WD")[str(uri_ref).replace(str(WD), '')]
    elif str(uri_ref).startswith(str(WDT)):
        return getattr(NamespaceWrapper, "WDT")[str(uri_ref).replace(str(WDT), '')]
    elif str(uri_ref).startswith(str(DDIS)):
        return getattr(NamespaceWrapper, "DDIS")[str(uri_ref).replace(str(DDIS), '')]
    elif str(uri_ref).startswith(str(RDFS)):
        return getattr(NamespaceWrapper, "RDFS")[str(uri_ref).replace(str(RDFS), '')]
    elif str(uri_ref).startswith(str(SCHEMA)):
        return getattr(NamespaceWrapper, "SCHEMA")[str(uri_ref).replace(str(SCHEMA), '')]
    else:
        return uri_ref