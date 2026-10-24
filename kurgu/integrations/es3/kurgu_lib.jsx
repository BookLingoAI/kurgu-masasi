// kurgu_lib.jsx - ES3-compatible helper library for Premiere Pro ExtendScript.
// Prepended to every command sent through the MCP Bridge. No JSON object assumed (ES3).
var KL = (function () {
  var TPS = 254016000000; // ticks per second

  function q(s) {
    s = String(s);
    var out = '"';
    for (var i = 0; i < s.length; i++) {
      var c = s.charAt(i), code = s.charCodeAt(i);
      if (c === '"') out += '\\"';
      else if (c === '\\') out += '\\\\';
      else if (c === '\n') out += '\\n';
      else if (c === '\r') out += '\\r';
      else if (c === '\t') out += '\\t';
      else if (code < 32 || code > 126) {
        var h = code.toString(16);
        while (h.length < 4) h = '0' + h;
        out += '\\u' + h;
      } else out += c;
    }
    return out + '"';
  }

  function ser(v) {
    if (v === null || v === undefined) return 'null';
    var t = typeof v;
    if (t === 'number') return isFinite(v) ? String(v) : 'null';
    if (t === 'boolean') return String(v);
    if (t === 'string') return q(v);
    if (v instanceof Array) {
      var a = [];
      for (var i = 0; i < v.length; i++) a.push(ser(v[i]));
      return '[' + a.join(',') + ']';
    }
    var o = [];
    for (var k in v) {
      if (v.hasOwnProperty(k)) o.push(q(k) + ':' + ser(v[k]));
    }
    return '{' + o.join(',') + '}';
  }

  function ok(data) { return ser({ success: true, data: data }); }
  function fail(msg) { return ser({ success: false, error: String(msg) }); }

  function secs(t) {
    try { return Number(t.ticks) / TPS; } catch (e) { return null; }
  }
  function timeFromSeconds(s) {
    var t = new Time();
    t.ticks = String(Math.round(s * TPS));
    return t;
  }
  function timeFromTicks(ticks) {
    var t = new Time();
    t.ticks = String(ticks);
    return t;
  }

  // ---- project items ---------------------------------------------------------------------
  function normPath(p) { return String(p).replace(/\//g, '\\').toLowerCase(); }

  function walk(bin, fn) {
    for (var i = 0; i < bin.children.numItems; i++) {
      var it = bin.children[i];
      if (fn(it) === true) return it;
      if (it.type === 2) { var r = walk(it, fn); if (r) return r; }
    }
    return null;
  }

  function findItemByPath(path) {
    var want = normPath(path);
    return walk(app.project.rootItem, function (it) {
      try { return it.type !== 2 && normPath(it.getMediaPath()) === want; } catch (e) { return false; }
    });
  }

  function findBin(name, parent) {
    parent = parent || app.project.rootItem;
    for (var i = 0; i < parent.children.numItems; i++) {
      var it = parent.children[i];
      if (it.type === 2 && it.name === name) return it;
    }
    return null;
  }

  function ensureBin(name, parent) {
    parent = parent || app.project.rootItem;
    return findBin(name, parent) || parent.createBin(name);
  }

  function importFiles(paths, binName) {
    var bin = binName ? ensureBin(binName) : app.project.rootItem;
    var need = [];
    for (var i = 0; i < paths.length; i++) if (!findItemByPath(paths[i])) need.push(paths[i]);
    if (need.length) {
      var okImp = app.project.importFiles(need, true, bin, false);
      if (!okImp) throw new Error('importFiles failed: ' + need.join(', '));
    }
    var items = [];
    for (var j = 0; j < paths.length; j++) {
      var it = findItemByPath(paths[j]);
      if (!it) throw new Error('imported item not found: ' + paths[j]);
      items.push(it);
    }
    return items;
  }

  function findSequence(name) {
    for (var i = 0; i < app.project.sequences.numSequences; i++) {
      var s = app.project.sequences[i];
      if (s.name === name) return s;
    }
    return null;
  }

  function projectInfo() {
    var p = app.project;
    var seqs = [];
    for (var i = 0; i < p.sequences.numSequences; i++) {
      var s = p.sequences[i];
      seqs.push({ name: s.name, id: s.sequenceID, width: s.frameSizeHorizontal, height: s.frameSizeVertical,
                  timebase: s.timebase, end: Number(s.end) / TPS });
    }
    var items = [];
    walk(p.rootItem, function (it) {
      var mp = '';
      try { mp = it.getMediaPath(); } catch (e) {}
      if (items.length < 500) items.push({ name: it.name, type: it.type, path: mp });
      return false;
    });
    return { version: app.version, name: p.name, path: p.path, sequences: seqs, items: items,
             active: p.activeSequence ? p.activeSequence.name : null };
  }

  function trackState(coll) {
    var out = [];
    for (var i = 0; i < coll.numTracks; i++) {
      var tr = coll[i], clips = [];
      for (var j = 0; j < tr.clips.numItems; j++) {
        var c = tr.clips[j], mp = '';
        try { mp = c.projectItem ? c.projectItem.getMediaPath() : ''; } catch (e) {}
        var dis = null;
        try { dis = c.disabled; } catch (e2) {}
        clips.push({ name: c.name, start: secs(c.start), end: secs(c.end), inPoint: secs(c.inPoint),
                     outPoint: secs(c.outPoint), path: mp, disabled: dis });
      }
      out.push({ index: i, name: tr.name, clips: clips });
    }
    return out;
  }

  function sequenceState(name) {
    var s = name ? findSequence(name) : app.project.activeSequence;
    if (!s) throw new Error('sequence not found: ' + name);
    var st = s.getSettings ? s.getSettings() : null;
    return { name: s.name, width: s.frameSizeHorizontal, height: s.frameSizeVertical,
             timebaseTicks: s.timebase, fps: s.timebase ? TPS / Number(s.timebase) : null,
             end: Number(s.end) / TPS,
             video: trackState(s.videoTracks), audio: trackState(s.audioTracks) };
  }

  // Create a sequence whose settings match the clip (keeps 4K60 etc.), then empty it.
  function sequenceFromClip(name, item, binName) {
    var existing = findSequence(name);
    if (existing) throw new Error('sequence already exists: ' + name);
    var bin = binName ? ensureBin(binName) : app.project.rootItem;
    var seq = app.project.createNewSequenceFromClips(name, [item], bin);
    if (!seq) throw new Error('createNewSequenceFromClips failed');
    clearSequence(seq);
    return seq;
  }

  function clearSequence(seq) {
    var colls = [seq.videoTracks, seq.audioTracks];
    for (var c = 0; c < colls.length; c++) {
      for (var i = 0; i < colls[c].numTracks; i++) {
        var tr = colls[c][i];
        for (var j = tr.clips.numItems - 1; j >= 0; j--) tr.clips[j].remove(false, false);
      }
    }
  }

  // Place [srcIn, srcOut) of item at timeline time `at` (seconds) on video track vIdx (+ linked audio).
  function placeSegment(seq, item, srcIn, srcOut, at, vIdx) {
    item.setInPoint(timeFromSeconds(srcIn), 4);
    item.setOutPoint(timeFromSeconds(srcOut), 4);
    seq.videoTracks[vIdx].overwriteClip(item, timeFromSeconds(at));
  }

  function resetInOut(item) {
    try { item.clearInPoint(4); item.clearOutPoint(4); } catch (e) {}
  }

  function exportFrame(seq, seconds, outPath) {
    app.enableQE();
    var qs = qe.project.getActiveSequence();
    if (!qs) throw new Error('no active QE sequence');
    var fps = TPS / Number(seq.timebase);
    var f = Math.round(seconds * fps);
    var hh = Math.floor(f / (fps * 3600)), mm = Math.floor(f / (fps * 60)) % 60,
        ss = Math.floor(f / fps) % 60, ff = f % Math.round(fps);
    function two(n) { return (n < 10 ? '0' : '') + n; }
    var tc = two(hh) + ':' + two(mm) + ':' + two(ss) + ':' + two(ff);
    qs.exportFramePNG(tc, outPath);
    return tc;
  }

  // ---- full edit helpers ------------------------------------------------------------------
  function clipAt(track, at) {
    var best = null, bd = 1e9;
    for (var i = 0; i < track.clips.numItems; i++) {
      var c = track.clips[i], d = Math.abs(Number(c.start.ticks) / TPS - at);
      if (d < bd) { bd = d; best = c; }
    }
    return bd < 0.05 ? best : null;
  }

  // place [srcIn, srcOut) of item on any video/audio track at `at`; returns the track item
  function placeOn(track, item, srcIn, srcOut, at) {
    item.setInPoint(timeFromSeconds(srcIn), 4);
    item.setOutPoint(timeFromSeconds(srcOut), 4);
    track.overwriteClip(item, timeFromSeconds(at));
    var c = clipAt(track, at);
    if (!c) throw new Error('clip not found after placing at ' + at);
    return c;
  }

  // still image for `dur` seconds
  function placeStill(track, item, at, dur) {
    try { item.setInPoint(timeFromSeconds(0), 4); item.setOutPoint(timeFromSeconds(dur), 4); } catch (e) {}
    track.overwriteClip(item, timeFromSeconds(at));
    var c = clipAt(track, at);
    if (!c) throw new Error('still not found after placing at ' + at);
    try { if (Math.abs(Number(c.end.ticks) / TPS - (at + dur)) > 0.02) c.end = timeFromSeconds(at + dur); } catch (e2) {}
    return c;
  }

  function comp(clip, matchName) {
    for (var i = 0; i < clip.components.numItems; i++) {
      var c = clip.components[i];
      if (c.matchName === matchName) return c;
    }
    return null;
  }

  // keyframe time for a timeline time t (keys are in source-media time of the clip)
  function keyTime(clip, t) {
    return timeFromSeconds(Number(clip.inPoint.ticks) / TPS + (t - Number(clip.start.ticks) / TPS));
  }

  var SEQ_W = 1920, SEQ_H = 1080;
  function setSeqSize(w, h) { SEQ_W = w; SEQ_H = h; }

  // position given in normalized frame coords (0..1); converts if Premiere uses pixels
  function posVal(prop, x, y) {
    var cur = prop.getValue();
    if (cur && cur.length && Math.abs(cur[0]) > 2) return [x * SEQ_W, y * SEQ_H];
    return [x, y];
  }

  function setMotion(clip, scalePct, x, y) {
    var m = comp(clip, 'AE.ADBE Motion');
    if (!m) throw new Error('Motion component not found');
    var pos = m.properties[0], sc = m.properties[1];
    sc.setValue(scalePct, true);
    pos.setValue(posVal(pos, x, y), true);
  }

  // keys: [[t, scalePct, x, y], ...] timeline seconds
  function motionKeys(clip, keys) {
    var m = comp(clip, 'AE.ADBE Motion');
    var pos = m.properties[0], sc = m.properties[1];
    pos.setTimeVarying(true); sc.setTimeVarying(true);
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i], kt = keyTime(clip, k[0]);
      sc.addKey(kt); sc.setValueAtKey(kt, k[1], true);
      pos.addKey(kt); pos.setValueAtKey(kt, posVal(pos, k[2], k[3]), true);
    }
  }

  // opacity fade in/out (seconds) at the clip edges
  function fade(clip, fin, fout) {
    var o = comp(clip, 'AE.ADBE Opacity');
    if (!o) return false;
    var op = o.properties[0];
    var t0 = Number(clip.start.ticks) / TPS, t1 = Number(clip.end.ticks) / TPS;
    op.setTimeVarying(true);
    var pts = [];
    if (fin > 0) { pts.push([t0, 0]); pts.push([t0 + fin, 100]); }
    if (fout > 0) { pts.push([t1 - fout, 100]); pts.push([t1 - 1 / 1000, 0]); }
    for (var i = 0; i < pts.length; i++) {
      var kt = keyTime(clip, pts[i][0]);
      op.addKey(kt); op.setValueAtKey(kt, pts[i][1], true);
    }
    return true;
  }

  function clearTrack(track) {
    for (var j = track.clips.numItems - 1; j >= 0; j--) track.clips[j].remove(false, false);
  }

  function ensureTracks(seq, nv, na) {
    if (seq.videoTracks.numTracks >= nv && seq.audioTracks.numTracks >= na) return true;
    app.enableQE();
    var qs = qe.project.getActiveSequence();
    var addV = Math.max(0, nv - seq.videoTracks.numTracks), addA = Math.max(0, na - seq.audioTracks.numTracks);
    qs.addTracks(addV, seq.videoTracks.numTracks, addA, 1, seq.audioTracks.numTracks, 0, 1, 0);
    if (seq.videoTracks.numTracks < nv || seq.audioTracks.numTracks < na)
      throw new Error('could not add tracks: ' + seq.videoTracks.numTracks + 'V/' + seq.audioTracks.numTracks + 'A');
    return true;
  }

  return {
    clipAt: clipAt, placeOn: placeOn, placeStill: placeStill, comp: comp, keyTime: keyTime, setSeqSize: setSeqSize,
    setMotion: setMotion, motionKeys: motionKeys, fade: fade, clearTrack: clearTrack, ensureTracks: ensureTracks,
    TPS: TPS, ser: ser, ok: ok, fail: fail, secs: secs, timeFromSeconds: timeFromSeconds,
    timeFromTicks: timeFromTicks, findItemByPath: findItemByPath, ensureBin: ensureBin, importFiles: importFiles,
    findSequence: findSequence, projectInfo: projectInfo, sequenceState: sequenceState,
    sequenceFromClip: sequenceFromClip, clearSequence: clearSequence, placeSegment: placeSegment,
    resetInOut: resetInOut, exportFrame: exportFrame
  };
})();
