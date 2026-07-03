"""Intake + acquisition. See :mod:`anchor.ingest.intake` and :mod:`anchor.ingest.scout`."""
from anchor.ingest.intake import probe, ingest, VideoMeta, ProbeError
from anchor.ingest.scout import Scope, Candidate, SourceConnector, LocalFolderConnector

__all__ = ["probe", "ingest", "VideoMeta", "ProbeError",
           "Scope", "Candidate", "SourceConnector", "LocalFolderConnector"]
