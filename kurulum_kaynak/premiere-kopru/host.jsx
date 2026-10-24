// Host-side ExtendScript (runs in Premiere Pro's ExtendScript engine)
// Dosya köprüsü Node.js olmadan da çalışsın diye dosya işlemleri burada yapılır.

function mcpBridgePing() {
  return "pong";
}

function mcpEnsureDir(dir) {
  var f = new Folder(dir);
  if (!f.exists) f.create();
  return f.exists ? "ok" : "fail";
}

// Klasördeki ilk cmd_*.jsx dosyasını bulur, okur, siler, çalıştırır ve res_*.json yazar.
// Dönen değer: "" (iş yok) veya işlenen dosyanın adı.
function mcpPollOnce(dir) {
  var folder = new Folder(dir);
  if (!folder.exists) return "";
  var files = folder.getFiles("cmd_*.jsx");
  if (!files || files.length === 0) return "";
  files.sort(function (a, b) { return a.name < b.name ? -1 : (a.name > b.name ? 1 : 0); });
  // only take files that are at least 1 s old: the writer may still be writing a fresh file
  var cmd = null, now = new Date().getTime();
  for (var fi = 0; fi < files.length; fi++) {
    var m = files[fi].modified;
    if (!m || now - m.getTime() >= 1000) { cmd = files[fi]; break; }
  }
  if (!cmd) return "";
  var name = decodeURI(cmd.name);
  cmd.encoding = "UTF-8";
  if (!cmd.open("r")) return "ERR:" + name;
  var script = cmd.read();
  cmd.close();
  cmd.remove();

  var id = name.replace(/^cmd_/, "").replace(/\.jsx$/, "");
  var result;
  try {
    result = eval(script);
    if (result === undefined || result === null) result = '{"success":true,"data":null}';
    result = String(result);
  } catch (e) {
    result = '{"success":false,"error":' + mcpQuote(String(e) + (e.line ? " (line " + e.line + ")" : "")) + "}";
  }
  var res = new File(folder.fsName + "/res_" + id + ".json");
  res.encoding = "UTF-8";
  res.open("w");
  res.write(result);
  res.close();
  return name;
}

function mcpQuote(s) {
  return '"' + s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\r/g, "\\r").replace(/\n/g, "\\n") + '"';
}

// Default bridge folder: first line of bridge_dir.txt next to this file.
function mcpDefaultDir() {
  try {
    var here = new File($.fileName).parent;
    var f = new File(here.fsName + "/bridge_dir.txt");
    if (!f.exists) return "";
    f.encoding = "UTF-8";
    f.open("r");
    var s = f.readln();
    f.close();
    return s ? s.replace(/^\s+|\s+$/g, "") : "";
  } catch (e) {
    return "";
  }
}
